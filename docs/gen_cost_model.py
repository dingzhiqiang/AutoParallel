"""Generate cost model flow diagram for AutoParallel."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, xy, w, h, text, fc, subtext=None, fs=9.5, stfs=7, tc="white",
        ec="#333", lw=1.3, alpha=1.0):
    x, y = xy
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                       facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=2)
    ax.add_patch(b)
    if subtext:
        ax.text(x + w/2, y + h/2 + 0.12, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)
        ax.text(x + w/2, y + h/2 - 0.15, subtext, ha="center", va="center",
                fontsize=stfs, color=tc, alpha=0.85, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)


def arr(ax, s, e, c="#555", lw=1.5, cs="arc3,rad=0.0"):
    a = FancyArrowPatch(s, e, arrowstyle="->", connectionstyle=cs,
                        color=c, linewidth=lw, zorder=1, mutation_scale=14)
    ax.add_patch(a)


# Colors
CB = "#1565C0"  # blue
CO = "#E65100"  # orange
CP = "#6A1B9A"  # purple
CT = "#00695C"  # teal
CR = "#B71C1C"  # red
CG = "#2E7D32"  # green
CGR = "#616161"  # gray

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(-0.5, 14)
ax.set_ylim(-0.5, 10.5)
ax.axis("off")
ax.set_aspect("equal")

ax.text(7, 10.2, "Training Cost Model Pipeline", ha="center",
        fontsize=15, fontweight="bold", color="#212121")
ax.text(7, 9.85, "Per-layer compute + communication → pipeline schedule → throughput score",
        ha="center", fontsize=9, color="#757575", style="italic")

# === Row 1: Per-layer Compute ===
ax.text(0.3, 9.2, "① Per-Layer Compute Cost", fontsize=10, fontweight="bold", color=CB)

box(ax, (0.3, 8.2), 3.2, 0.8, "Attention FLOPs", CB,
    subtext="MHA: 8H²T/t  |  MLA: proj + score")
box(ax, (4, 8.2), 3.2, 0.8, "FFN FLOPs", CB,
    subtext="Dense: 6Hd_intT/t  |  MoE: routed+shared")
box(ax, (7.7, 8.2), 3.2, 0.8, "T_compute", CB,
    subtext="F_total / F_peak (GPU TFLOPS)")

arr(ax, (3.5, 8.6), (4, 8.6), c=CB)
arr(ax, (7.2, 8.6), (7.7, 8.6), c=CB)

# Profiling override arrow
box(ax, (11.3, 8.2), 2.3, 0.8, "GEMM Profile", CR,
    subtext="measured time_us", tc="white")
arr(ax, (11.3, 8.6), (10.9, 8.6), c=CR, cs="arc3,rad=0.0")
ax.text(11.15, 8.6, "override", fontsize=7, color=CR, ha="right", style="italic")

# === Row 2: Per-layer Communication ===
ax.text(0.3, 7.5, "② Per-Layer Communication Cost (α-β model)", fontsize=10,
        fontweight="bold", color=CO)

box(ax, (0.3, 6.5), 3, 0.7, "TP AllReduce", CO,
    subtext="α_nv + V/B_nv (BW degrade)")
box(ax, (3.8, 6.5), 3.5, 0.7, "EP AllToAll", CO,
    subtext="Hierarchical: NVLink intra + IB inter")
box(ax, (7.8, 6.5), 2.8, 0.7, "CP Ring", CO,
    subtext="KV transfer: α(c-1) + V/B")
box(ax, (11.1, 6.5), 2.5, 0.7, "Comm Profile", CR,
    subtext="measured time_us", tc="white")
arr(ax, (11.1, 6.85), (10.6, 6.85), c=CR)

# === Row 3: Engine-aware combine ===
ax.text(0.3, 5.7, "③ Engine-Aware Per-Layer Time", fontsize=10,
        fontweight="bold", color=CP)

# Megatron path
box(ax, (0.3, 4.6), 5.5, 0.8, "Megatron: T_attn + T_tp + max(T_ep, T_ffn) + T_cp", CP,
    subtext="EP AllToAll overlaps with routed FFN compute")
# FSDP path
box(ax, (6.3, 4.6), 5.5, 0.8, "FSDP: T_compute + T_tp + T_ep + T_cp", CP,
    subtext="All communication serialized with compute")

arr(ax, (5, 6.5), (3, 5.45), c=CO, cs="arc3,rad=0.1")
arr(ax, (5, 6.5), (9, 5.45), c=CO, cs="arc3,rad=-0.1")

# x3 annotation
ax.text(12.2, 4.65, "× 3", fontsize=11, fontweight="bold", color=CP)
ax.text(12.2, 4.4, "(fwd=1x\nbwd=2x)", fontsize=7, color="#888", ha="center")

# === Row 4: Pipeline Schedule ===
ax.text(0.3, 3.9, "④ Pipeline Schedule (1F1B)", fontsize=10,
        fontweight="bold", color=CT)

box(ax, (0.3, 2.9), 5, 0.8, "T_stage = T_layer × ⌈L/p⌉ + T_pp_p2p", CT,
    subtext="Per pipeline stage time")
box(ax, (5.8, 2.9), 5.5, 0.8, "T_total = (p + n_mb - 1) × T_stage", CT,
    subtext="1F1B schedule, bubble = (p-1)/(p+n_mb-1)")

arr(ax, (3, 4.6), (2.8, 3.75), c=CP)
arr(ax, (9, 4.6), (8.5, 3.75), c=CP)
arr(ax, (5.3, 3.3), (5.8, 3.3), c=CT)

# Pipeline visual
# Draw 1F1B schedule mini-diagram
px, py = 11.6, 3.6
colors_1f1b = ["#42A5F5", "#42A5F5", "#42A5F5", "#EF5350", "#42A5F5", "#EF5350",
               "#42A5F5", "#EF5350", "#EF5350"]
for i, c in enumerate(colors_1f1b):
    rect = plt.Rectangle((px + i * 0.22, py), 0.2, 0.2, facecolor=c,
                          edgecolor="white", linewidth=0.5, zorder=2)
    ax.add_patch(rect)
ax.text(px + len(colors_1f1b)*0.22/2, py - 0.15, "F F F B F B F B B",
        fontsize=5.5, ha="center", color="#888", family="monospace")

# === Row 5: Output ===
ax.text(0.3, 2.2, "⑤ Throughput Score", fontsize=10, fontweight="bold", color=CG)

box(ax, (2.5, 1.2), 8, 0.8, "Score = d × n_mb / T_total", CG,
    subtext="Higher score = better strategy. Rank all feasible strategies by score.",
    fs=12)

arr(ax, (8.5, 2.9), (6.5, 2.05), c=CT)

# Legend
legend_items = [
    ("Compute Model", CB),
    ("Communication Model (α-β)", CO),
    ("Engine-Aware Fusion", CP),
    ("Pipeline Schedule", CT),
    ("Profiling Override (optional)", CR),
    ("Output Score", CG),
]
for i, (label, color) in enumerate(legend_items):
    yy = 1.0 - i * 0.3
    rect = plt.Rectangle((11.5, yy - 0.08), 0.25, 0.16, facecolor=color, zorder=2)
    ax.add_patch(rect)
    ax.text(11.85, yy, label, fontsize=7, va="center", color="#333")

plt.tight_layout()
fig.savefig("/tmp/AutoParallel/docs/images/cost_model.png", dpi=180,
            bbox_inches="tight", facecolor="white")
fig.savefig("/tmp/AutoParallel/docs/images/cost_model.svg",
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("cost_model done")
