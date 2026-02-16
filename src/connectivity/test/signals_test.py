"""
Python Translation of Professor's signals_test.m
=================================================
Creates synthetic test signals with KNOWN connectivity:

Ground Truth Network:
    Ch1 → Ch2 → Ch3 → Ch4
    Ch5: Pure noise (isolated)

This tests:
1. Can pipeline detect direct connections? (Ch1→Ch2, Ch2→Ch3, Ch3→Ch4)
2. Does PDC reject indirect paths? (Ch1→Ch3, Ch1→Ch4 should be LOW)
3. Can it isolate noise channels? (Ch5 should have ZERO connections)
"""

import numpy as np


def awgn(signal, snr_db):
    """
    Add White Gaussian Noise to achieve target SNR.
    
    Python equivalent of MATLAB's awgn(x, snr, 'measured', 'db')
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    snr_db : float
        Target Signal-to-Noise Ratio in dB
        
    Returns:
    --------
    noisy_signal : np.ndarray
        Signal with added noise
    """
    # Measure signal power
    signal_power = np.mean(signal**2)
    
    if signal_power < 1e-10:
        # Signal is essentially zero, just return it
        return signal
    
    # Convert SNR from dB to linear scale
    signal_power_db = 10 * np.log10(signal_power)
    noise_power_db = signal_power_db - snr_db
    noise_power = 10 ** (noise_power_db / 10)
    
    # Generate noise
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    
    return signal + noise


def signals_test(db_noise=20):
    """
    Generate test signals with cascade connectivity.
    
    EXACT Python translation of professor's signals_test.m
    
    Parameters:
    -----------
    db_noise : float
        SNR in dB (default: 20)
        
    Returns:
    --------
    x : np.ndarray
        Shape (1900, 5) - 5 channels, 1900 timepoints
        
    Ground Truth:
    -------------
    Ch1: Original 10 Hz cosine
    Ch2: Ch1 delayed by 2 samples + noise
    Ch3: Ch2 delayed by 2 samples + noise  
    Ch4: Ch3 delayed by 2 samples + noise
    Ch5: Pure noise (isolated)
    
    Expected Connectivity:
    ----------------------
    HIGH:  Ch1→Ch2, Ch2→Ch3, Ch3→Ch4 (direct)
    LOW:   Ch1→Ch3, Ch1→Ch4 (indirect, PDC should reject)
    ZERO:  Ch5↔anything (noise only)
    """
    # Initialize
    x = np.zeros((2000, 5))
    
    # Parameters
    A = 10.0                # Amplitude
    samples_delay = 2       # Time delay between channels
    start_point = 100       # Remove startup transient
    
    # Time periods
    period_1 = np.arange(0, 1100)
    
    # Frequency
    f1 = 10  # 10 Hz cosine
    
    # Generate original cosine signal
    t = np.arange(1, 1201)  # Match MATLAB 1:1200
    xs1 = A * np.cos(2 * np.pi * t * f1 / 1000)
    
    # =========================================================================
    # Channel 1: Original signal (with offset to avoid edge effects)
    # =========================================================================
    x[period_1, 0] = xs1[100 + period_1]
    x[:1100, 0] = awgn(x[:1100, 0], db_noise)
    
    # =========================================================================
    # Channel 2: Ch1 delayed by 2 samples
    # =========================================================================
    x[samples_delay:1100, 1] = x[0:1100-samples_delay, 0]
    x[:, 1] = awgn(x[:, 1], db_noise)
    
    # =========================================================================
    # Channel 3: Ch2 delayed by 2 samples
    # =========================================================================
    x[samples_delay:1100, 2] = x[0:1100-samples_delay, 1]
    x[:, 2] = awgn(x[:, 2], db_noise)
    
    # =========================================================================
    # Channel 4: Ch3 delayed by 2 samples
    # =========================================================================
    x[samples_delay:1100, 3] = x[0:1100-samples_delay, 2]
    x[:, 3] = awgn(x[:, 3], db_noise)
    
    # =========================================================================
    # Channel 5: Pure white noise (isolated)
    # =========================================================================
    x[:, 4] = 0.5 * np.random.randn(2000)
    
    # Remove startup transient
    x = x[start_point:, :]  # Shape: (1900, 5)
    
    return x


if __name__ == "__main__":
    # Test the function
    import matplotlib.pyplot as plt
    
    print("=" * 80)
    print("TESTING signals_test.m (Python Translation)")
    print("=" * 80)
    
    # Generate signals
    x = signals_test(db_noise=20)
    
    print(f"\nSignal shape: {x.shape}")
    print(f"Expected: (1900, 5)")
    print(f"\nGround Truth Network:")
    print("  Ch1 → Ch2 → Ch3 → Ch4")
    print("  Ch5: Noise (isolated)")
    
    # Plot
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    
    for i in range(5):
        axes[i].plot(x[:, i], linewidth=0.5)
        axes[i].set_ylabel(f'Ch{i+1}', fontsize=11, fontweight='bold')
        axes[i].grid(alpha=0.3)
        
        if i < 4:
            axes[i].set_title(f'Channel {i+1} → Channel {i+2}' if i < 3 else 'Channel 4',
                             fontsize=10)
        else:
            axes[i].set_title('Channel 5 (Noise only)', fontsize=10, color='red')
    
    axes[4].set_xlabel('Time (samples)', fontsize=11)
    fig.suptitle('signals_test.m - Cascade Network\nCh1→Ch2→Ch3→Ch4, Ch5=Noise',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('signals_test_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: signals_test_visualization.png")
    print("=" * 80)
