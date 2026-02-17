"""
signals_test3_19ch.py
=====================
Synthetic validation test with 19 channels - matching real EEG setup.

Designed to mirror:
  - 19 standard channels (like the TUC dataset)
  - 256 Hz sampling rate (like the TUC dataset)
  - 4-second epochs → 1024 samples (like step0_create_epochs_label.py)
  - MVAR order p=12 (like step2_compute_connectivity.py)

Ground Truth Network (randomly assigned, clearly defined):
----------------------------------------------------------
CONNECTED CHANNELS (known directed links):
  Hub:      Ch1 (Fp1)  → Ch4 (F3), Ch9 (C3), Ch14 (P3)   [long-range broadcaster]
  Chain A:  Ch4 (F3)   → Ch8 (T3) → Ch13 (T5)             [frontal→temporal cascade]
  Chain B:  Ch9 (C3)   → Ch11 (C4)                         [central connection]
  Bridge:   Ch8 (T3)   → Ch14 (P3)                         [temporal→parietal]

ISOLATED CHANNELS (pure noise, no true connections):
  Ch2  (Fp2), Ch3  (F7),  Ch5  (Fz),  Ch6  (F4),
  Ch7  (F8),  Ch10 (Cz),  Ch12 (T4),  Ch15 (Pz),
  Ch16 (P4),  Ch17 (T6),  Ch18 (O1),  Ch19 (O2)

Expected PDC result:
  HIGH  : Ch1→Ch4, Ch1→Ch9, Ch1→Ch14, Ch4→Ch8, Ch8→Ch13, Ch9→Ch11, Ch8→Ch14
  LOW   : Indirect paths (e.g. Ch1→Ch13 via Ch4→Ch8)
  ~ZERO : Any connection involving noise channels (Ch2,3,5,6,7,10,12,15,16,17,18,19)

Why this matters:
  - Tests pipeline at REAL scale (19 channels, not 5)
  - Tests that noise channels are correctly isolated even when surrounded by
    active channels (harder problem than in the 5-channel tests)
  - Uses same fs=256 Hz and epoch_length=1024 samples as TUC pipeline

Usage:
  python signals_test3_19ch.py
  # Or import and use in validate_connectivity_19ch.py
"""

import numpy as np


# =============================================================================
# CHANNEL DEFINITIONS  (matching TUC standard montage from step0)
# =============================================================================

CHANNELS = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',   # 0-4
    'F4',  'F8',  'T3',  'C3',  'Cz',   # 5-9
    'C4',  'T4',  'T5',  'P3',  'Pz',   # 10-14
    'P4',  'T6',  'O1',  'O2'            # 15-18
]

N_CHANNELS = 19
FS = 256          # Hz  – matches TUC dataset
EPOCH_SAMPLES = 1024  # 4 seconds at 256 Hz

# =============================================================================
# GROUND TRUTH CONNECTIVITY MAP
# (index pairs: source → target, 0-based)
# =============================================================================

# Hub channel: Fp1 (index 0)
# Chain A:     Fp1→F3(3), F3→T3(7), T3→T5(12)
# Long-range:  Fp1→C3(8), Fp1→P3(13)
# Central:     C3(8)→C4(10)
# Bridge:      T3(7)→P3(13)

GROUND_TRUTH_CONNECTIONS = [
    (0,  3),   # Fp1  → F3
    (0,  8),   # Fp1  → C3
    (0,  13),  # Fp1  → P3
    (3,  7),   # F3   → T3
    (7,  12),  # T3   → T5
    (8,  10),  # C3   → C4
    (7,  13),  # T3   → P3
]

# Noise-only channels (no true outgoing or incoming connections)
NOISE_CHANNELS = [1, 2, 4, 5, 6, 9, 11, 14, 15, 16, 17, 18]


# =============================================================================
# HELPER: AWGN
# =============================================================================

def awgn(signal, snr_db):
    """Add White Gaussian Noise to reach a target SNR (same as test1 & test2)."""
    signal_power = np.mean(signal ** 2)
    if signal_power < 1e-10:
        return signal
    signal_power_db = 10 * np.log10(signal_power)
    noise_power_db  = signal_power_db - snr_db
    noise_power     = 10 ** (noise_power_db / 10)
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

def signals_test3_19ch(db_noise=20, seed=None):
    """
    Generate 19-channel synthetic EEG with a known connectivity structure.

    Parameters
    ----------
    db_noise : float
        SNR in dB for the noise added to each driven channel (default: 20).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    x : np.ndarray, shape (EPOCH_SAMPLES, 19)
        Synthetic signals, one column per channel, ready to be used as a
        single epoch (transpose before feeding to VAR: VAR(x) directly,
        or x.T if your pipeline expects (channels, samples)).

    ground_truth_connections : list of (src, tgt) tuples
        The true directed connections (0-based indices).

    noise_channels : list of int
        Channel indices that contain only noise.
    """
    if seed is not None:
        np.random.seed(seed)

    # -----------------------------------------------------------------
    # 1. Build a longer buffer then trim to remove startup transients
    # -----------------------------------------------------------------
    BUFFER   = 512          # extra samples to discard at start
    N_TOTAL  = EPOCH_SAMPLES + BUFFER

    x = np.zeros((N_TOTAL, N_CHANNELS))

    # -----------------------------------------------------------------
    # 2. Source signal: 10 Hz cosine at 256 Hz  (same style as test1/2)
    # -----------------------------------------------------------------
    t    = np.arange(N_TOTAL)
    f1   = 10.0             # Hz
    src  = 10.0 * np.cos(2 * np.pi * f1 * t / FS)

    # -----------------------------------------------------------------
    # 3. Place source in hub channel (Fp1, index 0)
    # -----------------------------------------------------------------
    x[:, 0] = awgn(src, db_noise)

    # -----------------------------------------------------------------
    # 4. Build driven channels with 1-sample delay (at 256 Hz this is
    #    ~4 ms, a realistic short conduction delay)
    # -----------------------------------------------------------------
    DELAY = 1   # samples

    def drive(src_col, tgt_col, delay=DELAY):
        """Copy src → tgt with a delay and add noise."""
        x[delay:, tgt_col] = x[:N_TOTAL - delay, src_col]
        x[:, tgt_col] = awgn(x[:, tgt_col], db_noise)

    # Hub  → F3 (index 3)
    drive(0, 3)
    # Hub  → C3 (index 8)
    drive(0, 8)
    # Hub  → P3 (index 13)
    drive(0, 13)

    # F3   → T3 (index 7)
    drive(3, 7)
    # T3   → T5 (index 12)
    drive(7, 12)

    # C3   → C4 (index 10)
    drive(8, 10)

    # T3   → P3 (index 13)  – P3 already receives from hub; this ADDS to it
    x[DELAY:, 13] += x[:N_TOTAL - DELAY, 7]   # additive drive (no extra awgn needed)

    # -----------------------------------------------------------------
    # 5. Noise-only channels (completely independent)
    # -----------------------------------------------------------------
    noise_amplitude = 0.5
    for ch in NOISE_CHANNELS:
        x[:, ch] = noise_amplitude * np.random.randn(N_TOTAL)

    # -----------------------------------------------------------------
    # 6. Trim startup buffer → final epoch of 1024 samples
    # -----------------------------------------------------------------
    x = x[BUFFER:, :]   # shape: (1024, 19)

    assert x.shape == (EPOCH_SAMPLES, N_CHANNELS), \
        f"Expected ({EPOCH_SAMPLES}, {N_CHANNELS}), got {x.shape}"

    return x, GROUND_TRUTH_CONNECTIONS, NOISE_CHANNELS


# =============================================================================
# QUICK SELF-TEST  (run as standalone script)
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 80)
    print("signals_test3_19ch.py – 19-Channel Synthetic Validation Signal")
    print("=" * 80)

    x, gt_conn, noise_ch = signals_test3_19ch(db_noise=20, seed=42)

    print(f"\nSignal shape : {x.shape}  (samples × channels)")
    print(f"Sampling rate: {FS} Hz")
    print(f"Duration     : {EPOCH_SAMPLES / FS:.1f} s  ({EPOCH_SAMPLES} samples)\n")

    print("Ground-truth connections (source → target):")
    for src, tgt in gt_conn:
        print(f"  {CHANNELS[src]:4s} (ch{src+1:02d}) → {CHANNELS[tgt]:4s} (ch{tgt+1:02d})")

    print(f"\nNoise-only channels ({len(noise_ch)} total):")
    print("  " + ", ".join(f"{CHANNELS[c]} (ch{c+1})" for c in noise_ch))

    # ----------------------------------------------------------------
    # Plot: show all 19 channels in a grid, colour-code by role
    # ----------------------------------------------------------------
    active_ch  = sorted(set(range(N_CHANNELS)) - set(noise_ch))
    time_axis  = np.arange(EPOCH_SAMPLES) / FS

    fig, axes = plt.subplots(19, 1, figsize=(14, 22), sharex=True)
    fig.suptitle(
        "19-Channel Synthetic Test Signal (signals_test3_19ch)\n"
        "Blue = connected / driven,  Red = noise only",
        fontsize=13, fontweight="bold"
    )

    for i in range(N_CHANNELS):
        colour = "steelblue" if i in active_ch else "crimson"
        label  = CHANNELS[i]
        if i in active_ch:
            label += " ✓"
        else:
            label += " (noise)"
        axes[i].plot(time_axis, x[:, i], linewidth=0.6, color=colour)
        axes[i].set_ylabel(label, fontsize=8, rotation=0, labelpad=55)
        axes[i].tick_params(labelleft=False)
        axes[i].grid(alpha=0.2)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()
    plt.savefig("signals_test3_19ch_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✅ Saved: signals_test3_19ch_visualization.png")

    # ----------------------------------------------------------------
    # Also show the ground-truth adjacency matrix
    # ----------------------------------------------------------------
    adj = np.zeros((N_CHANNELS, N_CHANNELS))
    for src, tgt in gt_conn:
        adj[tgt, src] = 1.0   # convention: rows=target, cols=source (matches PDC)

    fig2, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(adj, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(N_CHANNELS))
    ax.set_yticks(range(N_CHANNELS))
    ax.set_xticklabels(CHANNELS, rotation=90, fontsize=8)
    ax.set_yticklabels(CHANNELS, fontsize=8)
    ax.set_xlabel("Source (FROM)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Target (TO)",   fontsize=11, fontweight="bold")
    ax.set_title("Ground-Truth Adjacency Matrix\n(red = true directed connection)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("signals_test3_19ch_ground_truth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved: signals_test3_19ch_ground_truth.png")

    print("\n" + "=" * 80)
    print("Next step: run validate_connectivity_19ch.py to check pipeline accuracy")
    print("=" * 80)