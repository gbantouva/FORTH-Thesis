"""
validate_connectivity_19ch.py
==============================
Validates your DTF/PDC pipeline at FULL 19-channel scale.

This completes the validation suite:
  test1 → 5-ch cascade     (signals_test.py)
  test2 → 5-ch branching   (signals_test2.py)
  test3 → 19-ch mixed      (signals_test3_19ch.py)  ← THIS FILE

Run with:
    python validate_connectivity_19ch.py

Ground Truth Network (see signals_test3_19ch.py for full details):
    Hub:     Fp1 → F3, C3, P3
    Chain A: F3  → T3 → T5
    Central: C3  → C4
    Bridge:  T3  → P3
    Noise:   Fp2, F7, Fz, F4, F8, Cz, T4, Pz, P4, T6, O1, O2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import linalg
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
warnings.filterwarnings("ignore")

from signals_test3 import (
    signals_test3_19ch,
    CHANNELS,
    GROUND_TRUTH_CONNECTIONS,
    NOISE_CHANNELS,
    N_CHANNELS,
    FS,
    EPOCH_SAMPLES
)


# =============================================================================
# CONNECTIVITY FUNCTIONS  (same as step2_compute_connectivity.py)
# =============================================================================

def compute_dtf_pdc_from_var(coefs, fs=256.0, nfft=512):
    p, K, _ = coefs.shape
    n_freqs  = nfft // 2 + 1
    freqs    = np.linspace(0, fs / 2, n_freqs)

    A_f = np.zeros((n_freqs, K, K), dtype=complex)
    H_f = np.zeros((n_freqs, K, K), dtype=complex)
    I   = np.eye(K)

    for f_idx, f in enumerate(freqs):
        A_sum = np.zeros((K, K), dtype=complex)
        for k in range(p):
            phase    = np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
            A_sum   += coefs[k] * phase
        A_f[f_idx]   = I - A_sum
        try:
            H_f[f_idx] = linalg.inv(A_f[f_idx])
        except linalg.LinAlgError:
            H_f[f_idx] = linalg.pinv(A_f[f_idx])

    # PDC (column-wise normalisation)
    pdc = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        Af        = A_f[f_idx]
        col_norms = np.sqrt(np.sum(np.abs(Af) ** 2, axis=0))
        col_norms[col_norms == 0] = 1e-10
        pdc[:, :, f_idx] = np.abs(Af) / col_norms[None, :]

    # DTF (row-wise normalisation)
    dtf = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        Hf        = H_f[f_idx]
        row_norms = np.sqrt(np.sum(np.abs(Hf) ** 2, axis=1))
        row_norms[row_norms == 0] = 1e-10
        dtf[:, :, f_idx] = np.abs(Hf) / row_norms[:, None]

    return dtf, pdc, freqs


def process_epoch(data, fs=FS, fixed_order=12, nfft=512):
    """
    Run MVAR + DTF/PDC on one epoch.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
    """
    data_std = np.std(data)
    if data_std < 1e-10:
        return None

    data_scaled = data / data_std

    try:
        model   = VAR(data_scaled.T)          # statsmodels expects (samples, channels)
        results = model.fit(maxlags=fixed_order, trend='c', verbose=False)
    except Exception:
        return None

    if results.k_ar == 0:
        return None

    dtf_spec, pdc_spec, freqs = compute_dtf_pdc_from_var(results.coefs, fs, nfft)

    # Integrate over 0.5–45 Hz (passband of TUC data)
    mask            = (freqs >= 0.5) & (freqs <= 45.0)
    dtf_integrated  = np.mean(dtf_spec[:, :, mask], axis=2)
    pdc_integrated  = np.mean(pdc_spec[:, :, mask], axis=2)

    np.fill_diagonal(dtf_integrated, 0.0)
    np.fill_diagonal(pdc_integrated, 0.0)

    return {'dtf': dtf_integrated, 'pdc': pdc_integrated}


# =============================================================================
# RUN VALIDATION
# =============================================================================

def run_validation(snr_db=20, n_trials=10):
    """
    Average DTF/PDC over n_trials independent realisations and check
    whether the ground-truth connections are recovered.
    """
    print(f"\n{'='*80}")
    print(f"19-CHANNEL VALIDATION  –  SNR = {snr_db} dB  ({n_trials} trials)")
    print(f"{'='*80}")

    pdc_list = []
    dtf_list = []

    for trial in range(n_trials):
        x, _, _ = signals_test3_19ch(db_noise=snr_db, seed=trial)
        epoch   = x.T          # (19, 1024) — channels × samples
        result  = process_epoch(epoch, fs=FS, fixed_order=12)
        if result is not None:
            pdc_list.append(result['pdc'])
            dtf_list.append(result['dtf'])

    if not pdc_list:
        print("❌  Pipeline FAILED on all trials!")
        return None, None, False

    pdc_avg = np.mean(pdc_list, axis=0)   # (19, 19)
    dtf_avg = np.mean(dtf_list, axis=0)

    print(f"\n✅  Succeeded on {len(pdc_list)}/{n_trials} trials\n")

    # ------------------------------------------------------------------
    # Print key values
    # ------------------------------------------------------------------
    print("PDC values for GROUND-TRUTH connections (should be HIGH):")
    gt_pdc_values = []
    for src, tgt in GROUND_TRUTH_CONNECTIONS:
        val = pdc_avg[tgt, src]
        gt_pdc_values.append(val)
        print(f"  {CHANNELS[src]:4s} → {CHANNELS[tgt]:4s}  :  {val:.3f}")

    print("\nPDC values for NOISE channels (should be LOW):")
    noise_pdc_values = []
    for ch in NOISE_CHANNELS:
        out_mean = np.mean([pdc_avg[j, ch] for j in range(N_CHANNELS) if j != ch])
        in_mean  = np.mean([pdc_avg[ch, j] for j in range(N_CHANNELS) if j != ch])
        avg      = (out_mean + in_mean) / 2
        noise_pdc_values.append(avg)
        print(f"  {CHANNELS[ch]:4s}  avg connectivity  :  {avg:.3f}")

    # ------------------------------------------------------------------
    # Validation checks
    # ------------------------------------------------------------------
    print(f"\n{'-'*80}")
    print("VALIDATION CHECKS")
    print(f"{'-'*80}")

    passed = 0
    total  = 0

    mean_gt    = np.mean(gt_pdc_values)
    mean_noise = np.mean(noise_pdc_values)

    # 1. Ground-truth connections are detected
    total += 1
    if mean_gt > 0.25:
        print(f"✅  Mean GT PDC > 0.25       : {mean_gt:.3f}")
        passed += 1
    else:
        print(f"❌  Mean GT PDC too low       : {mean_gt:.3f}")

    # 2. Noise channels have lower connectivity than GT channels
    total += 1
    if mean_gt > mean_noise * 1.5:
        print(f"✅  GT PDC > 1.5× noise PDC  : {mean_gt:.3f} vs {mean_noise:.3f}")
        passed += 1
    else:
        print(f"❌  GT not clearly > noise   : {mean_gt:.3f} vs {mean_noise:.3f}")

    # 3. Every individual GT connection is above noise floor
    total += 1
    min_gt = min(gt_pdc_values)
    if min_gt > mean_noise:
        print(f"✅  Min GT PDC > noise mean  : {min_gt:.3f} vs {mean_noise:.3f}")
        passed += 1
    else:
        print(f"❌  Some GT connections weak  : min={min_gt:.3f} noise={mean_noise:.3f}")

    # 4. Hub channel (Fp1) has highest out-degree
    total += 1
    hub_out = np.sum(pdc_avg[:, 0]) - pdc_avg[0, 0]
    max_out = max(np.sum(pdc_avg[:, ch]) - pdc_avg[ch, ch] for ch in range(N_CHANNELS))
    if hub_out == max_out or hub_out > 0.9 * max_out:
        print(f"✅  Hub (Fp1) is top source   : out-strength={hub_out:.3f}")
        passed += 1
    else:
        print(f"⚠️   Hub out-strength {hub_out:.3f} (max in data={max_out:.3f})")

    print(f"\nPassed {passed}/{total} checks")
    overall_pass = passed >= 3    # require at least 3/4

    if overall_pass:
        print("✅  19-CHANNEL VALIDATION PASSED")
    else:
        print("❌  19-CHANNEL VALIDATION FAILED")

    return pdc_avg, dtf_avg, overall_pass


# =============================================================================
# PLOT
# =============================================================================

def plot_results(pdc_avg, dtf_avg, snr_db, output_dir):
    """
    Three-panel figure:
      Left   – Ground-truth adjacency matrix
      Middle – Recovered PDC
      Right  – Difference (PDC - GT): shows false positives / missed connections
    """
    # Build ground-truth matrix (rows=target, cols=source, like PDC)
    gt_matrix = np.zeros((N_CHANNELS, N_CHANNELS))
    for src, tgt in GROUND_TRUTH_CONNECTIONS:
        gt_matrix[tgt, src] = 1.0

    diff = pdc_avg - gt_matrix   # positive = false positive, negative = missed

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    kw = dict(xticklabels=CHANNELS, yticklabels=CHANNELS, square=True,
              linewidths=0.3, linecolor='gray')

    # Ground truth
    sns.heatmap(gt_matrix, ax=axes[0], cmap='Reds', vmin=0, vmax=1,
                annot=False, cbar_kws={'label': 'True connection'}, **kw)
    axes[0].set_title('Ground Truth Adjacency\n(1 = true directed connection)',
                      fontweight='bold')
    axes[0].set_xlabel('Source (FROM)')
    axes[0].set_ylabel('Target (TO)')
    axes[0].tick_params(axis='x', rotation=90, labelsize=7)
    axes[0].tick_params(axis='y', rotation=0,  labelsize=7)

    # Recovered PDC
    sns.heatmap(pdc_avg, ax=axes[1], cmap='YlOrRd', vmin=0, vmax=pdc_avg.max(),
                annot=False, cbar_kws={'label': 'Mean PDC'}, **kw)

    # Highlight ground-truth connections with blue boxes
    for src, tgt in GROUND_TRUTH_CONNECTIONS:
        axes[1].add_patch(plt.Rectangle((src, tgt), 1, 1, fill=False,
                                        edgecolor='blue', lw=2.5))
    # Highlight noise channels with grey shading on axis ticks
    axes[1].set_title(f'Recovered PDC (SNR={snr_db} dB)\nBlue box = ground-truth connection',
                      fontweight='bold')
    axes[1].set_xlabel('Source (FROM)')
    axes[1].set_ylabel('Target (TO)')
    axes[1].tick_params(axis='x', rotation=90, labelsize=7)
    axes[1].tick_params(axis='y', rotation=0,  labelsize=7)

    # Colour-code tick labels: blue=active, red=noise
    for ax in [axes[1], axes[0]]:
        for tick, label in zip(ax.get_xticklabels(), CHANNELS):
            idx = CHANNELS.index(label)
            tick.set_color('crimson' if idx in NOISE_CHANNELS else 'steelblue')
        for tick, label in zip(ax.get_yticklabels(), CHANNELS):
            idx = CHANNELS.index(label)
            tick.set_color('crimson' if idx in NOISE_CHANNELS else 'steelblue')

    # Difference matrix
    vmax_diff = max(abs(diff.min()), abs(diff.max()), 0.1)
    sns.heatmap(diff, ax=axes[2], cmap='RdBu_r', center=0,
                vmin=-vmax_diff, vmax=vmax_diff,
                annot=False, cbar_kws={'label': 'PDC − GT'}, **kw)
    axes[2].set_title('Difference (PDC − Ground Truth)\nRed=false positive, Blue=missed',
                      fontweight='bold')
    axes[2].set_xlabel('Source (FROM)')
    axes[2].set_ylabel('Target (TO)')
    axes[2].tick_params(axis='x', rotation=90, labelsize=7)
    axes[2].tick_params(axis='y', rotation=0,  labelsize=7)

    # Legend for tick colour
    legend_handles = [
        mpatches.Patch(color='steelblue', label='Connected channel'),
        mpatches.Patch(color='crimson',   label='Noise-only channel'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.03))

    plt.suptitle(
        '19-Channel Connectivity Validation\n'
        f'Hub: Fp1 → F3, C3, P3  |  Chain: F3→T3→T5  |  Central: C3→C4  |  Bridge: T3→P3',
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    out_path = output_dir / f'validation_19ch_PDC_snr{snr_db}dB.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅  Saved: {out_path}")


def plot_connection_summary(pdc_avg, snr_db, output_dir):
    """
    Bar chart comparing PDC values of GT connections vs noise-channel connections.
    Clearest way to show the pipeline is working.
    """
    # GT connection values
    gt_labels = [f"{CHANNELS[s]}→{CHANNELS[t]}" for s, t in GROUND_TRUTH_CONNECTIONS]
    gt_vals   = [pdc_avg[t, s] for s, t in GROUND_TRUTH_CONNECTIONS]

    # Noise channel values (mean absolute connectivity per channel)
    noise_labels = [CHANNELS[c] for c in NOISE_CHANNELS]
    noise_vals   = [
        np.mean([pdc_avg[j, c] for j in range(N_CHANNELS) if j != c] +
                [pdc_avg[c, j] for j in range(N_CHANNELS) if j != c])
        for c in NOISE_CHANNELS
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # GT connections
    colors_gt = ['#2ecc71' if v > np.mean(noise_vals) else '#e74c3c' for v in gt_vals]
    bars1 = axes[0].bar(range(len(gt_labels)), gt_vals, color=colors_gt,
                        alpha=0.85, edgecolor='black', linewidth=1.2)
    axes[0].axhline(np.mean(noise_vals), color='red', linestyle='--', linewidth=2,
                    label=f'Noise mean = {np.mean(noise_vals):.3f}')
    axes[0].set_xticks(range(len(gt_labels)))
    axes[0].set_xticklabels(gt_labels, rotation=35, ha='right', fontsize=9)
    axes[0].set_ylabel('Mean PDC', fontsize=11)
    axes[0].set_title('Ground-Truth Connections\n(green = above noise floor)',
                      fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3, axis='y')

    for bar, val in zip(bars1, gt_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                     f'{val:.3f}', ha='center', fontsize=8)

    # Noise channels
    axes[1].bar(range(len(noise_labels)), noise_vals, color='#95a5a6',
                alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[1].axhline(np.mean(gt_vals), color='green', linestyle='--', linewidth=2,
                    label=f'GT mean = {np.mean(gt_vals):.3f}')
    axes[1].set_xticks(range(len(noise_labels)))
    axes[1].set_xticklabels(noise_labels, rotation=35, ha='right', fontsize=9)
    axes[1].set_ylabel('Mean PDC', fontsize=11)
    axes[1].set_title('Noise-Only Channels\n(should be much lower than GT)',
                      fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3, axis='y')

    plt.suptitle(
        f'19-Channel Validation Summary  (SNR={snr_db} dB, p=12, fs=256 Hz)\n'
        f'GT mean = {np.mean(gt_vals):.3f}   vs   Noise mean = {np.mean(noise_vals):.3f}',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    out_path = output_dir / f'validation_19ch_summary_snr{snr_db}dB.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅  Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = Path("validation_results_19ch")
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("19-CHANNEL CONNECTIVITY PIPELINE VALIDATION")
    print("=" * 80)
    print("Matching TUC dataset: 19 channels, fs=256 Hz, epoch=1024 samples, p=12")
    print("=" * 80)

    all_passed = True

    for snr in [20, 10]:
        print(f"\n\n{'#'*80}")
        print(f"# SNR = {snr} dB")
        print(f"{'#'*80}")

        pdc_avg, dtf_avg, passed = run_validation(snr_db=snr, n_trials=10)

        if not passed:
            all_passed = False

        if pdc_avg is not None:
            plot_results(pdc_avg, dtf_avg, snr, output_dir)
            plot_connection_summary(pdc_avg, snr, output_dir)

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    if all_passed:
        print("🎉  19-CHANNEL VALIDATION PASSED!")
        print("    Your pipeline correctly identifies directed connections at real EEG scale.")
        print("    Noise channels are suppressed as expected.")
        print("    You can trust the DTF/PDC results on the TUC dataset.")
    else:
        print("❌  19-CHANNEL VALIDATION FAILED")
        print("    Common causes:")
        print("    1. Wrong transpose: check VAR(data_scaled.T) in process_epoch()")
        print("    2. Wrong fs: make sure fs=256 matches your data")
        print("    3. Wrong integration band: check 0.5–45 Hz mask")
    print("=" * 80)


if __name__ == "__main__":
    main()