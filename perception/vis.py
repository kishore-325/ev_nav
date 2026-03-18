"""
Generate a visual architecture diagram of OrigUNet.
Run: python3 -m perception.vis
Output: perception/plots/architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

# Colors
C_INPUT = "#A8D5BA"
C_ENC = "#7EB3D8"
C_BOTTLE = "#E8A87C"
C_DEC = "#D4A5C9"
C_OUTPUT = "#F7DC6F"
C_SKIP = "#C0C0C0"


def draw_block(ax, x, y, w, h, label, sublabel, color, fontsize=8):
    """Draw a rounded rectangle block with label."""
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.05", facecolor=color,
        edgecolor="black", linewidth=1.2
    )
    ax.add_patch(rect)
    ax.text(x, y + 0.15, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold")
    ax.text(x, y - 0.2, sublabel, ha="center", va="center",
            fontsize=6.5, color="#333333", style="italic")
    return (x, y)


def draw_arrow(ax, start, end, color="black", lw=1.2, style="->"):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style, color=color,
        lw=lw, mutation_scale=12,
        connectionstyle="arc3,rad=0"
    )
    ax.add_patch(arrow)


def draw_skip_arrow(ax, start, end, color="#888888"):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="->", color=color,
        lw=1.0, mutation_scale=10, linestyle="dashed",
        connectionstyle="arc3,rad=-0.3"
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 11)
    ax.set_aspect("equal")
    ax.axis("off")

    bw, bh = 2.2, 0.8  # block width, height

    # === Title ===
    ax.text(6, 10.3, "OrigUNet Architecture", ha="center", va="center",
            fontsize=16, fontweight="bold")
    ax.text(6, 9.85, "Event Camera Log-Diff → Depth Estimation", ha="center",
            va="center", fontsize=10, color="#555555")

    # === Input ===
    p_input = draw_block(ax, 2, 9, bw, bh,
                         "Input", "(N, 1, 260, 346)", C_INPUT)

    # === _form_input ===
    p_form = draw_block(ax, 2, 8, bw, bh,
                        "_form_input", "(N, 2, 260, 346)", C_INPUT)

    # === Encoder blocks (left column, going down) ===
    enc_x = 2
    enc_specs = [
        ("Enc Block 1", "2→32, 3×3 conv ×2\n(N, 32, 256, 342)", 6.8),
        ("MaxPool 2×2", "(N, 32, 128, 171)", 5.8),
        ("Enc Block 2", "32→64, 3×3 conv ×2\n(N, 64, 124, 167)", 4.8),
        ("MaxPool 2×2", "(N, 64, 62, 83)", 3.8),
        ("Enc Block 3", "64→128, 3×3 conv ×2\n(N, 128, 58, 79)", 2.8),
        ("MaxPool 2×2", "(N, 128, 29, 39)", 1.8),
        ("Enc Block 4", "128→256, 3×3 conv ×2\n(N, 256, 25, 35)", 0.8),
        ("MaxPool 2×2", "(N, 256, 12, 17)", -0.2),
    ]

    enc_positions = []
    for label, sub, yy in enc_specs:
        if "MaxPool" in label:
            p = draw_block(ax, enc_x, yy, bw, 0.55, label, sub, "#B0C4DE", fontsize=7)
        else:
            p = draw_block(ax, enc_x, yy, bw, bh, label, sub, C_ENC, fontsize=8)
        enc_positions.append(p)

    # === Bottleneck ===
    p_bottle = draw_block(ax, 6, -0.2, 2.4, bh,
                          "Bottleneck", "256→512, 3×3 conv ×2\n(N, 512, 8, 13)", C_BOTTLE)

    # === Decoder blocks (right column, going up) ===
    dec_x = 10
    dec_specs = [
        ("UpConv 512→256", "(N, 256, 16, 26)", 0.8),
        ("Dec Block 1", "512→256, 3×3 conv ×2\n(N, 256, 12, 22)", 1.8),
        ("UpConv 256→128", "(N, 128, 24, 44)", 2.8),
        ("Dec Block 2", "256→128, 3×3 conv ×2\n(N, 128, 20, 40)", 3.8),
        ("UpConv 128→64", "(N, 64, 40, 80)", 4.8),
        ("Dec Block 3", "128→64, 3×3 conv ×2\n(N, 64, 36, 76)", 5.8),
        ("UpConv 64→32", "(N, 32, 72, 152)", 6.8),
        ("Dec Block 4", "64→32, 3×3 conv ×2\n(N, 32, 68, 148)", 7.8),
    ]

    dec_positions = []
    for label, sub, yy in dec_specs:
        if "UpConv" in label:
            p = draw_block(ax, dec_x, yy, bw, 0.55, label, sub, "#E6C8E0", fontsize=7)
        else:
            p = draw_block(ax, dec_x, yy, bw, bh, label, sub, C_DEC, fontsize=8)
        dec_positions.append(p)

    # === Output head ===
    p_outconv = draw_block(ax, 10, 8.6, bw, 0.65,
                           "1×1 Conv → 1ch", "(N, 1, 68, 148)", C_OUTPUT, fontsize=8)
    p_upsample = draw_block(ax, 10, 9.4, bw, 0.55,
                            "Bilinear Upsample", "(N, 1, 260, 346)", C_OUTPUT, fontsize=7)
    p_sigmoid = draw_block(ax, 10, 10.0, bw, 0.45,
                           "Sigmoid", "depth ∈ [0, 1]", C_OUTPUT, fontsize=8)

    # === Vertical arrows — encoder ===
    draw_arrow(ax, (2, 9 - bh / 2), (2, 8 + bh / 2))
    draw_arrow(ax, (2, 8 - bh / 2), (2, 6.8 + bh / 2))
    for i in range(len(enc_specs) - 1):
        y1 = enc_specs[i][2] - (bh / 2 if "MaxPool" not in enc_specs[i][0] else 0.275)
        y2 = enc_specs[i + 1][2] + (bh / 2 if "MaxPool" not in enc_specs[i + 1][0] else 0.275)
        draw_arrow(ax, (enc_x, y1), (enc_x, y2))

    # Encoder → Bottleneck
    draw_arrow(ax, (enc_x, -0.2 - 0.275), (6 - 1.2, -0.2))

    # Bottleneck → Decoder
    draw_arrow(ax, (6 + 1.2, -0.2), (dec_x, 0.8 - 0.275))

    # === Vertical arrows — decoder ===
    for i in range(len(dec_specs) - 1):
        y1 = dec_specs[i][2] + (bh / 2 if "Dec Block" in dec_specs[i][0] else 0.275)
        y2 = dec_specs[i + 1][2] - (bh / 2 if "Dec Block" in dec_specs[i + 1][0] else 0.275)
        draw_arrow(ax, (dec_x, y1), (dec_x, y2))

    # Decoder → Output
    draw_arrow(ax, (dec_x, 7.8 + bh / 2), (dec_x, 8.6 - 0.325))
    draw_arrow(ax, (dec_x, 8.6 + 0.325), (dec_x, 9.4 - 0.275))
    draw_arrow(ax, (dec_x, 9.4 + 0.275), (dec_x, 10.0 - 0.225))

    # === Skip connections (dashed, encoder → decoder) ===
    skip_pairs = [
        # (enc_block_y, dec_block_y, label)
        (6.8, 7.8, "e1 → d4\ncrop(72,152)"),   # Enc1 → Dec4
        (4.8, 5.8, "e2 → d3\ncrop(40,80)"),     # Enc2 → Dec3
        (2.8, 3.8, "e3 → d2\ncrop(24,44)"),     # Enc3 → Dec2
        (0.8, 1.8, "e4 → d1\ncrop(16,26)"),     # Enc4 → Dec1
    ]

    for enc_y, dec_y, label in skip_pairs:
        mid_y = (enc_y + dec_y) / 2
        start = (enc_x + bw / 2, enc_y)
        end = (dec_x - bw / 2, dec_y)
        draw_skip_arrow(ax, start, end, color="#888888")
        # Label on the skip arrow
        mid_x = (enc_x + dec_x) / 2
        ax.text(mid_x, mid_y + 0.55, label, ha="center", va="center",
                fontsize=6, color="#666666", style="italic")

    # === Legend ===
    legend_items = [
        (C_INPUT, "Input / Preprocessing"),
        (C_ENC, "Encoder (Conv+ReLU)"),
        ("#B0C4DE", "MaxPool 2×2"),
        (C_BOTTLE, "Bottleneck"),
        ("#E6C8E0", "Transposed Conv (Upsample)"),
        (C_DEC, "Decoder (Conv+ReLU)"),
        (C_OUTPUT, "Output Head"),
    ]
    for i, (color, label) in enumerate(legend_items):
        lx, ly = 0.0, -0.5 - i * 0.35
        # We'll skip if out of bounds; adjust ylim
    # Extend ylim for legend
    ax.set_ylim(-3.5, 10.8)
    for i, (color, label) in enumerate(legend_items):
        lx, ly = 0.3, -1.0 - i * 0.4
        rect = mpatches.FancyBboxPatch(
            (lx - 0.15, ly - 0.12), 0.3, 0.24,
            boxstyle="round,pad=0.02", facecolor=color,
            edgecolor="black", linewidth=0.8
        )
        ax.add_patch(rect)
        ax.text(lx + 0.35, ly, label, ha="left", va="center", fontsize=7)

    # Dashed line for skip connection legend
    ly_skip = -1.0 - len(legend_items) * 0.4
    ax.plot([0.15, 0.45], [ly_skip, ly_skip], linestyle="dashed",
            color="#888888", lw=1.0)
    ax.text(0.65, ly_skip, "Skip Connection (centre crop + concat)",
            ha="left", va="center", fontsize=7, color="#666666")

    # === Note ===
    ax.text(6, -3.2,
            "All conv layers use 3×3 kernels (no padding) + ReLU. "
            "Final output bilinearly upsampled from 68×148 → 260×346.",
            ha="center", va="center", fontsize=7.5, color="#444444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC"))

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "architecture.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
