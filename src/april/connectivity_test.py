import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

# ── Parameters ────────────────────────────────────────────────────────
K = 5
T = 2000
p = 2
fs = 256
channels = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5']

# ── Simulate MVAR: CH1->CH2->CH3->CH5, CH4 standalone ────────────────
A1 = np.zeros((K, K))
A1[1, 0] = 0.8   # CH1 -> CH2
A1[2, 1] = 0.8   # CH2 -> CH3
A1[4, 2] = 0.8   # CH3 -> CH5
for i in range(K):
    A1[i, i] = 0.3   # self-feedback for stability

A2 = np.zeros((K, K))
A2[1, 0] = 0.1
A2[2, 1] = 0.1
A2[4, 2] = 0.1

X = np.zeros((K, T))
for t in range(p, T):
    X[:, t] = (A1 @ X[:, t-1] + A2 @ X[:, t-2]
               + 0.5 * np.random.randn(K))

# ── Fit MVAR via OLS ──────────────────────────────────────────────────
n = T - p
Z = np.zeros((n, K * p))
Y = X[:, p:].T
for k in range(p):
    Z[:, k*K:(k+1)*K] = X[:, p-1-k:T-1-k].T

B_hat = np.linalg.lstsq(Z, Y, rcond=None)[0]
A_hat = [B_hat[k*K:(k+1)*K, :].T for k in range(p)]

# ── Compute A(f) and H(f) ─────────────────────────────────────────────
NFFT = 256

def compute_Af_Hf(A_hat, p, K, NFFT, fs):
    Af = np.zeros((NFFT, K, K), dtype=complex)
    for fi in range(NFFT):
        f = fi * fs / NFFT
        Af_fi = np.eye(K, dtype=complex)
        for k in range(p):
            Af_fi -= A_hat[k] * np.exp(-1j * 2 * np.pi * f * (k+1) / fs)
        Af[fi] = Af_fi
    Hf = np.array([np.linalg.inv(Af[fi]) for fi in range(NFFT)])
    return Af, Hf

print("Computing A(f) and H(f)...")
Af, Hf = compute_Af_Hf(A_hat, p, K, NFFT, fs)

# ── PDC: column-normalized from A(f) ─────────────────────────────────
PDC = np.zeros((NFFT, K, K))
for fi in range(NFFT):
    for j in range(K):
        col_norm = np.sqrt(np.sum(np.abs(Af[fi, :, j])**2))
        if col_norm > 1e-10:
            PDC[fi, :, j] = np.abs(Af[fi, :, j]) / col_norm

# ── DTF Kaminski: row-normalized from H(f) ───────────────────────────
DTF_row = np.zeros((NFFT, K, K))
for fi in range(NFFT):
    for i in range(K):
        row_norm = np.sqrt(np.sum(np.abs(Hf[fi, i, :])**2))
        if row_norm > 1e-10:
            DTF_row[fi, i, :] = np.abs(Hf[fi, i, :]) / row_norm

# ── DTF Alexandra: column-normalized from H(f) ───────────────────────
DTF_col = np.zeros((NFFT, K, K))
for fi in range(NFFT):
    for j in range(K):
        col_norm = np.sqrt(np.sum(np.abs(Hf[fi, :, j])**2))
        if col_norm > 1e-10:
            DTF_col[fi, :, j] = np.abs(Hf[fi, :, j]) / col_norm

# ── Broadband average + zero diagonal ────────────────────────────────
PDC_bb     = PDC.mean(axis=0);     np.fill_diagonal(PDC_bb, 0)
DTF_row_bb = DTF_row.mean(axis=0); np.fill_diagonal(DTF_row_bb, 0)
DTF_col_bb = DTF_col.mean(axis=0); np.fill_diagonal(DTF_col_bb, 0)

# ── Print matrices to console ─────────────────────────────────────────
def print_matrix(M, name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    header = f"{'':8s}" + "".join(f"  {c:>5s}" for c in channels)
    print(header)
    print("-"*60)
    for i, ch in enumerate(channels):
        row_str = f"  {ch:6s}" + "".join(f"  {M[i,j]:5.3f}" for j in range(K))
        print(row_str)
    print()
    print("  Row sums (sum over sources, per sink):")
    for i, ch in enumerate(channels):
        print(f"    {ch}: {M[i,:].sum():.4f}")
    print("  Column sums (sum over sinks, per source):")
    for j, ch in enumerate(channels):
        print(f"    {ch}: {M[:,j].sum():.4f}")

print_matrix(PDC_bb,     "PDC  — column-norm, A(f)  [Baccala & Sameshima 2001]")
print_matrix(DTF_row_bb, "DTF  — row-norm,    H(f)  [Kaminski & Blinowska 1991]")
print_matrix(DTF_col_bb, "DTF* — column-norm, H(f)  [Alexandra/Tsipouraki variant]")

# ── Ground truth annotation ───────────────────────────────────────────
print("\n" + "="*60)
print("  GROUND TRUTH")
print("="*60)
print("  True causal chain: CH1 -> CH2 -> CH3 -> CH5")
print("  CH4: standalone (no connections)")
print()
print("  Expected strong off-diagonal cells:")
print("    [CH2, CH1] = CH1 -> CH2  (direct)")
print("    [CH3, CH2] = CH2 -> CH3  (direct)")
print("    [CH5, CH3] = CH3 -> CH5  (direct)")
print()
print("  DTF (both variants) should also show indirect paths:")
print("    [CH3, CH1] via CH2 (1-hop indirect)")
print("    [CH5, CH1] via CH2->CH3 (2-hop indirect)")
print("    [CH5, CH2] via CH3 (1-hop indirect)")
print()
print("  PDC should suppress all indirect paths.")

# ── Plot heatmaps ─────────────────────────────────────────────────────
matrices = [
    (PDC_bb,     "PDC\n[Baccala & Sameshima 2001]\nColumn-norm | A(f) | Direct only"),
    (DTF_row_bb, "DTF Kaminski\n[Kaminski & Blinowska 1991]\nRow-norm | H(f) | Sink-centric"),
    (DTF_col_bb, "DTF*  Alexandra variant\n[Tsipouraki 2024]\nColumn-norm | H(f) | Source-centric"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "PDC vs DTF vs DTF*  —  5-Channel Simulated MVAR Test\n"
    "True structure: CH1 → CH2 → CH3 → CH5   |   CH4 standalone",
    fontsize=13, fontweight='bold', y=1.02
)

for ax, (M, title) in zip(axes, matrices):
    im = ax.imshow(M, cmap='YlOrRd_r', vmin=0, vmax=0.7, aspect='auto')
    ax.set_xticks(range(K)); ax.set_xticklabels(channels)
    ax.set_yticks(range(K)); ax.set_yticklabels(channels)
    ax.set_xlabel("Source (j)", fontsize=11)
    ax.set_ylabel("Sink (i)", fontsize=11)
    ax.set_title(title, fontsize=10, pad=10)

    # Annotate cell values
    for i in range(K):
        for j in range(K):
            val = M[i, j]
            color = 'white' if val > 0.45 else 'black'
            ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold' if val > 0.3 else 'normal')

    # Highlight ground truth cells
    for (gi, gj) in [(1,0), (2,1), (4,2)]:
        ax.add_patch(plt.Rectangle((gj-0.5, gi-0.5), 1, 1,
                                   fill=False, edgecolor='blue',
                                   linewidth=2.5, linestyle='--'))

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Add legend note
fig.text(0.5, -0.03,
         "Blue dashed box = ground truth direct connection   "
         "|   Color scale: dark red = high connectivity, yellow = low",
         ha='center', fontsize=9, style='italic', color='#444')

plt.tight_layout()
plt.savefig("connectivity_test_5ch.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as: connectivity_test_5ch.png")
