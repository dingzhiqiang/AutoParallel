"""Generate inference model diagram for AutoParallel."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, xy, w, h, text, fc, subtext=None, fs=9.5, stfs=7, tc="white",
        ec="#333", lw=1.3):
    x, y = xy
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(b)
    if subtext:
        ax.text(x + w/2, y + h/2 + 0.13, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)
        ax.text(x + w/2, y + h/2 - 0.17, subtext, ha="center", va="center",
                fontsize=stfs, color=tc, alpha=0.85, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)


def arr(ax, s, e, c="#555", lw=1.5, cs="arc3,rad=0.0"):
    a = FancyArrowPatch(s, e, arrowstyle="->", connectionstyle=cs,
                        color=c, linewidth=lw, zorder=1, mutation_scale=14)
    ax.add_patch(a)


# Colors
CB = "#1565C0"   # blue - prefill
CO = "#E65100"   # orange - decode
CG = "#2E7D32"   # green - output
CT = "#00695C"   # teal
CP = "#6A1B9A"   # purple
CGR = "#757575"

fig, ax = plt.subplots(figsize=(14, 8.5))
ax.set_xlim(-0.5, 14)
ax.set_ylim(-0.5, 9)
ax.axis("off")
ax.set_aspect("equal")

ax.text(7, 8.7, "Inference Performance Model", ha="center",
        fontsize=15, fontweight="bold", color="#212121")
ax.text(7, 8.35, "Roofline model: Prefill (compute-bound) vs Decode (memory-bandwidth-bound)",
        ha="center", fontsize=9, color=CGR, style="italic")

# === LEFT: Prefill ===
# Background
pfill_bg = FancyBboxPatch((0.2, 2.2), 6, 5.6, boxstyle="round,pad=0.12",
                           facecolor="#E3F2FD", edgecolor="#1565C0",
                           linewidth=1.5, linestyle="--", zorder=0, alpha=0.5)
ax.add_patch(pfill_bg)
ax.text(3.2, 7.55, "PREFILL PHASE", ha="center", fontsize=11,
        fontweight="bold", color=CB)
ax.text(3.2, 7.2, "Compute-Bound", ha="center", fontsize=9,
        color=CB, style="italic")

box(ax, (0.5, 6.1), 5.4, 0.8, "Compute: F(T_prefill) / F_peak", CB,
    subtext="T_prefill = ISL × batch / TP")
box(ax, (0.5, 5.0), 5.4, 0.8, "Communication: T_tp + T_ep", CB,
    subtext="TP AllReduce + EP AllToAll per layer")
box(ax, (0.5, 3.8), 5.4, 0.8, "Per-Layer Time = Compute + Comm", "#1976D2",
    subtext="× ⌈L/p⌉ layers + PP P2P overhead")
box(ax, (0.5, 2.5), 5.4, 0.8, "Prefill TPS = ISL × batch / T_total", "#0D47A1",
    subtext="Tokens processed per second")

arr(ax, (3.2, 6.1), (3.2, 5.85), c=CB)
arr(ax, (3.2, 5.0), (3.2, 4.65), c=CB)
arr(ax, (3.2, 3.8), (3.2, 3.35), c=CB)

# Compute-bound indicator
ax.annotate("", xy=(6.1, 6.5), xytext=(6.1, 5.4),
            arrowprops=dict(arrowstyle="<->", color=CB, lw=1.2))
ax.text(6.3, 5.95, "compute\ndominates", fontsize=6.5, color=CB,
        ha="left", va="center")

# === RIGHT: Decode ===
dec_bg = FancyBboxPatch((7.2, 2.2), 6.3, 5.6, boxstyle="round,pad=0.12",
                         facecolor="#FBE9E7", edgecolor="#E65100",
                         linewidth=1.5, linestyle="--", zorder=0, alpha=0.5)
ax.add_patch(dec_bg)
ax.text(10.35, 7.55, "DECODE PHASE", ha="center", fontsize=11,
        fontweight="bold", color=CO)
ax.text(10.35, 7.2, "Memory-Bandwidth-Bound", ha="center", fontsize=9,
        color=CO, style="italic")

box(ax, (7.5, 6.1), 5.7, 0.8, "Compute: F(batch) / F_peak", CO,
    subtext="Very small (batch tokens only)")
box(ax, (7.5, 5.0), 5.7, 0.8, "Memory Read: (W + KV) / B_hbm", CO,
    subtext="Weights + KV cache from HBM")
box(ax, (7.5, 3.8), 5.7, 0.8, "Roofline: max(Compute, MemRead) + Comm", "#F4511E",
    subtext="× PP stages serial + P2P overhead")
box(ax, (7.5, 2.5), 5.7, 0.8, "Decode TPS = batch / T_decode", "#BF360C",
    subtext="Tokens generated per second")

arr(ax, (10.35, 6.1), (10.35, 5.85), c=CO)
arr(ax, (10.35, 5.0), (10.35, 4.65), c=CO)
arr(ax, (10.35, 3.8), (10.35, 3.35), c=CO)

# Memory-bound indicator
ax.annotate("", xy=(13.4, 6.5), xytext=(13.4, 5.4),
            arrowprops=dict(arrowstyle="<->", color=CO, lw=1.2))
ax.text(13.55, 5.95, "memory\ndominates", fontsize=6.5, color=CO,
        ha="left", va="center")

# === Bottom: Aggregate ===
box(ax, (3, 0.8), 8, 0.9, "Aggregate TPS = n_instances × Decode TPS", CG,
    subtext="Multi-instance deployment for maximum throughput. n_instances = N_gpu / (TP × PP)",
    fs=11)

arr(ax, (3.2, 2.5), (5.5, 1.75), c=CB, cs="arc3,rad=0.15")
arr(ax, (10.35, 2.5), (8.5, 1.75), c=CO, cs="arc3,rad=-0.15")

# Roofline mini chart
rx, ry = 0.5, 0.2
ax.text(rx + 1.0, ry + 0.45, "Roofline Model", fontsize=7, fontweight="bold",
        color=CGR, ha="center")
# Draw simple roofline shape
xs = [rx, rx + 0.7, rx + 2.0]
ys = [ry, ry + 0.35, ry + 0.35]
ax.plot(xs, ys, color="#F44336", linewidth=2, zorder=3)
ax.text(rx + 0.2, ry - 0.05, "mem-bound", fontsize=5.5, color=CO)
ax.text(rx + 1.4, ry - 0.05, "compute-bound", fontsize=5.5, color=CB)
ax.axvline(x=rx + 0.7, ymin=0.02, ymax=0.06, color="#999", linewidth=0.8,
           linestyle=":", zorder=1)
ax.text(rx + 0.7, ry - 0.15, "ridge point", fontsize=5, color="#999", ha="center")

# KV cache detail
box(ax, (11.5, 0.5), 2.2, 0.55, "KV Cache", "#78909C",
    subtext="batch × S̄ × L × kv_bytes / TP", fs=8, stfs=6)

plt.tight_layout()
fig.savefig("/tmp/AutoParallel/docs/images/inference_model.png", dpi=180,
            bbox_inches="tight", facecolor="white")
fig.savefig("/tmp/AutoParallel/docs/images/inference_model.svg",
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("inference_model done")
