"""Original architecture diagrams of the implemented models (matplotlib only).

These replace manuscript Figures 3.3-3.5, which were reproduced from other publications
and did not match the implemented networks.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_ENC, C_DEC, C_BOT, C_Q, C_ATT, C_IO = "#cfe0f7", "#d9f0d3", "#fde5c8", "#f6cfe3", "#fff3b0", "#e6e6e6"


def _box(ax, x, y, w, h, text, color, fs=8):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#444444", lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, p, q, style="-|>", ls="-", color="#444444"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, lw=0.9, ls=ls, color=color))


def unet_diagram(path, title, filters, attention=False, subtitle=""):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10.15)
    ax.set_ylim(-0.3, 5.5)
    ax.axis("off")
    sizes = [256, 128, 64, 32, 16]
    ys = [4.4, 3.4, 2.4, 1.4, 0.4]
    enc_x, dec_x = 1.75, 8.25
    _box(ax, 0.3, 4.4, 0.55, 0.5, "Input\n256²×1", C_IO, 7)
    for i in range(4):
        _box(ax, enc_x, ys[i], 1.9, 0.55, f"2×(Conv3×3–BN–ReLU)\n{sizes[i]}×{sizes[i]}×{filters[i]}", C_ENC, 7)
        _box(ax, dec_x, ys[i], 1.9, 0.55, f"2×(Conv3×3–BN–ReLU)\n{sizes[i]}×{sizes[i]}×{filters[i]}", C_DEC, 7)
        if i < 3:
            _arrow(ax, (enc_x, ys[i] - 0.28), (enc_x, ys[i + 1] + 0.28))
            _arrow(ax, (dec_x, ys[i + 1] + 0.28), (dec_x, ys[i] - 0.28))
        else:
            _arrow(ax, (enc_x, ys[i] - 0.28), (4.05, 0.45))
            _arrow(ax, (5.95, 0.45), (dec_x, ys[i] - 0.28))
        if attention:
            _box(ax, 5.0, ys[i], 1.0, 0.45, "Attention\ngate", C_ATT, 7)
            _arrow(ax, (enc_x + 0.95, ys[i]), (4.5, ys[i]), ls="--")
            _arrow(ax, (5.5, ys[i]), (dec_x - 0.95, ys[i]), ls="--")
        else:
            _arrow(ax, (enc_x + 0.95, ys[i]), (dec_x - 0.95, ys[i]), ls="--")
    _box(ax, 5.0, 0.4, 1.9, 0.55, f"Bottleneck\n16×16×{filters[4]}", C_BOT, 7)
    _arrow(ax, (0.58, 4.4), (enc_x - 0.95, 4.4))
    _box(ax, 9.72, 4.4, 0.55, 0.5, "1×1 conv\nsigmoid", C_IO, 7)
    _arrow(ax, (dec_x + 0.95, 4.4), (9.44, 4.4))
    ax.text(0.2, 5.3, title, fontsize=11, weight="bold")
    ax.text(0.2, 5.02, subtitle, fontsize=8, color="#444444")
    ax.text(0.2, 0.05, "down: 2×2 max-pool", fontsize=7, color="#444444")
    ax.text(7.6, 0.05, "up: 2×2 transposed conv", fontsize=7, color="#444444")
    ax.text(3.2, 4.72, "skip connection (concatenate)" + (" via attention gate" if attention else ""),
            fontsize=7, color="#444444")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def hybrid_diagram(path):
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10.15)
    ax.set_ylim(-1.9, 5.3)
    ax.axis("off")
    filters, sizes, ys = [32, 64, 128, 256], [256, 128, 64, 32], [4.4, 3.4, 2.4, 1.4]
    enc_x, dec_x = 1.75, 8.25
    _box(ax, 0.3, 4.4, 0.55, 0.5, "Input\n256²×1", C_IO, 7)
    _arrow(ax, (0.58, 4.4), (enc_x - 0.95, 4.4))
    for i in range(4):
        _box(ax, enc_x, ys[i], 1.9, 0.55, f"2×(Conv–BN–ReLU)+Dropout\n{sizes[i]}×{sizes[i]}×{filters[i]}", C_ENC, 6.5)
        _box(ax, dec_x, ys[i], 1.9, 0.55, f"2×(Conv–BN–ReLU)+Dropout\n{sizes[i]}×{sizes[i]}×{filters[i]}", C_DEC, 6.5)
        _arrow(ax, (enc_x + 0.95, ys[i]), (dec_x - 0.95, ys[i]), ls="--")
        if i < 3:
            _arrow(ax, (enc_x, ys[i] - 0.28), (enc_x, ys[i + 1] + 0.28))
            _arrow(ax, (dec_x, ys[i + 1] + 0.28), (dec_x, ys[i] - 0.28))
    _box(ax, enc_x, 0.4, 1.9, 0.55, "Bottleneck conv block\n16×16×512", C_BOT, 7)
    _arrow(ax, (enc_x, 1.12), (enc_x, 0.68))
    chain = [(1.75, -0.75, "Global average\npooling → 512"), (3.3, -0.75, "Dense(4, tanh)"),
             (5.0, -0.75, "4-qubit circuit\n(24 trainable)"), (6.7, -0.75, "Dense(512, ReLU)"),
             (8.25, -0.75, "Dense(16·16·512)\n→ reshape")]
    for j, (x, y, t) in enumerate(chain):
        _box(ax, x, y, 1.45, 0.6, t, C_Q if j == 2 else C_BOT, 7)
        if j:
            _arrow(ax, (chain[j - 1][0] + 0.73, y), (x - 0.73, y))
    _arrow(ax, (enc_x, 0.12), (enc_x, -0.45))
    _arrow(ax, (dec_x, -0.45), (dec_x, 1.12))
    _box(ax, 9.72, 4.4, 0.55, 0.5, "1×1 conv\nsigmoid", C_IO, 7)
    _arrow(ax, (dec_x + 0.95, 4.4), (9.44, 4.4))
    ax.text(0.2, 5.15, "Hybrid Quantum U-Net (as implemented)", fontsize=11, weight="bold")
    ax.text(0.2, 4.85, "Decoder up-sampling: 2× nearest-neighbour UpSampling2D; dashed = skip connections",
            fontsize=8, color="#444444")
    ax.text(3.0, -1.55, "Circuit: RY(π/2·xᵢ) encoding → 2 × [Rot(φ,θ,ω) on each qubit + CNOT ring] → ⟨Zᵢ⟩, i = 1…4.\n"
            "Ablations replace the circuit with Dense(4, tanh) or remove it.", fontsize=7.5, color="#444444")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_all(out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    hybrid_diagram(os.path.join(out_dir, "fig3_3_hybrid_quantum_unet.png"))
    unet_diagram(os.path.join(out_dir, "fig3_4_classical_unet.png"), "Classical U-Net (as implemented)",
                 [64, 128, 256, 512, 1024])
    unet_diagram(os.path.join(out_dir, "fig3_5_attention_unet.png"), "Attention U-Net (as implemented)",
                 [16, 32, 64, 128, 256], attention=True,
                 subtitle="Attention gate: 1×1 conv on gating and skip features, ReLU, 1×1 conv, sigmoid → multiply skip")
