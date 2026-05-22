"""Generate professional architecture diagrams for AutoParallel."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_rounded_box(ax, xy, width, height, text, facecolor, edgecolor="#333333",
                     fontsize=10, fontweight="bold", textcolor="white", alpha=1.0,
                     linestyle="-", linewidth=1.5, subtext=None, subtextsize=7.5):
    x, y = xy
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.08",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=linewidth, alpha=alpha, linestyle=linestyle,
                         zorder=2)
    ax.add_patch(box)
    if subtext:
        ax.text(x + width / 2, y + height / 2 + 0.15, text,
                ha="center", va="center", fontsize=fontsize, fontweight=fontweight,
                color=textcolor, zorder=3)
        ax.text(x + width / 2, y + height / 2 - 0.2, subtext,
                ha="center", va="center", fontsize=subtextsize, fontweight="normal",
                color=textcolor, alpha=0.85, zorder=3)
    else:
        ax.text(x + width / 2, y + height / 2, text,
                ha="center", va="center", fontsize=fontsize, fontweight=fontweight,
                color=textcolor, zorder=3)
    return box


def draw_arrow(ax, start, end, color="#555555", linewidth=1.8, style="->",
               connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch(start, end, arrowstyle=style,
                            connectionstyle=connectionstyle,
                            color=color, linewidth=linewidth, zorder=1,
                            mutation_scale=15)
    ax.add_patch(arrow)


# ============================================================
# Figure 1: System Architecture
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 9))
ax.set_xlim(-0.5, 13.5)
ax.set_ylim(-0.5, 9.5)
ax.axis("off")
ax.set_aspect("equal")

# Colors
C_INPUT = "#2196F3"      # blue - inputs
C_CORE = "#FF6F00"       # orange - core engine
C_COST = "#7B1FA2"       # purple - cost model
C_MEM = "#00897B"        # teal - memory model
C_PROF = "#C62828"       # red - profiling
C_OUT = "#2E7D32"        # green - output
C_BG = "#F5F5F5"         # light gray background

# Title
ax.text(6.5, 9.1, "AutoParallel System Architecture", ha="center", va="center",
        fontsize=16, fontweight="bold", color="#212121")

# --- Input Layer ---
ax.text(1.5, 8.5, "INPUT", ha="center", fontsize=9, fontweight="bold", color="#666")
draw_rounded_box(ax, (0, 7.5), 3, 0.75, "Model Config", C_INPUT,
                 subtext="HF config.json / Manual")
draw_rounded_box(ax, (3.5, 7.5), 3, 0.75, "Cluster Spec", C_INPUT,
                 subtext="N GPUs, GPU type, topology")
draw_rounded_box(ax, (7, 7.5), 3, 0.75, "Engine Preset", C_INPUT,
                 subtext="Megatron / FSDP / SGLang")
draw_rounded_box(ax, (10.5, 7.5), 2.5, 0.75, "Workload", C_INPUT,
                 subtext="batch, seq_len, mode")

# --- Core Engine (big box) ---
# Background rectangle
bg = FancyBboxPatch((0.3, 1.8), 12.8, 5.2, boxstyle="round,pad=0.15",
                    facecolor=C_BG, edgecolor="#BDBDBD", linewidth=1.5,
                    linestyle="--", zorder=0)
ax.add_patch(bg)
ax.text(6.8, 6.75, "Strategy Search Engine", ha="center", fontsize=12,
        fontweight="bold", color="#424242")

# Strategy Enumeration
draw_rounded_box(ax, (0.8, 5.5), 4, 0.9, "Strategy Enumeration", C_CORE,
                 subtext="DP × PP × TP × CP × EP constraint filtering")

# Arrows from inputs down
for x_start in [1.5, 5.0, 8.5, 11.75]:
    draw_arrow(ax, (x_start, 7.5), (x_start, 7.0), color="#2196F3")
# Merge arrows to enumeration
draw_arrow(ax, (3.0, 7.0), (2.8, 6.45), color="#888")
draw_arrow(ax, (5.0, 7.0), (3.5, 6.45), color="#888")
draw_arrow(ax, (8.5, 7.0), (4.0, 6.45), color="#888")
draw_arrow(ax, (11.75, 7.0), (4.5, 6.45), color="#888")

# Memory Estimation
draw_rounded_box(ax, (0.8, 3.8), 4, 0.9, "Memory Estimation", C_MEM,
                 subtext="Model / Gradient / Optimizer / Activation / KV Cache")
draw_arrow(ax, (2.8, 5.5), (2.8, 4.75), color=C_CORE)

# Cost Model (throughput)
draw_rounded_box(ax, (5.5, 3.8), 4, 0.9, "Throughput Cost Model", C_COST,
                 subtext="Compute (roofline) + Communication (α-β)")
draw_arrow(ax, (2.8, 3.8), (5.5, 4.25), color=C_MEM,
           connectionstyle="arc3,rad=-0.1")

# OOM filter
draw_rounded_box(ax, (5.5, 5.5), 3, 0.9, "OOM Filter", C_MEM,
                 subtext="GPU mem ≤ HBM, CPU mem ≤ Host")
draw_arrow(ax, (4.8, 5.95), (5.5, 5.95), color=C_CORE)

# Profiling (optional)
draw_rounded_box(ax, (10, 3.8), 3, 0.9, "Profiling Data", C_PROF,
                 subtext="GEMM + Comm interpolation",
                 textcolor="white")
ax.text(12.5, 3.65, "(optional)", ha="center", fontsize=7,
        color="#C62828", style="italic")
draw_arrow(ax, (10, 4.25), (9.5, 4.25), color=C_PROF, style="->",
           connectionstyle="arc3,rad=0.0")

# Ranking
draw_rounded_box(ax, (3.5, 2.2), 5, 0.9, "Rank by Efficiency Score", C_CORE,
                 subtext="Sort feasible strategies, generate pros/cons analysis")
draw_arrow(ax, (7.5, 3.8), (6.0, 3.15), color=C_COST)

# --- Profiling subsystem ---
draw_rounded_box(ax, (10, 5.5), 3, 0.9, "Hardware Profiler", C_PROF,
                 subtext="Slurm / Ray / Local backends")
draw_arrow(ax, (11.5, 5.5), (11.5, 4.75), color=C_PROF)
# Connect OOM filter to ranking
draw_arrow(ax, (7.0, 5.5), (6.0, 3.15), color=C_MEM,
           connectionstyle="arc3,rad=0.15")

# --- Output Layer ---
draw_rounded_box(ax, (1, 0.5), 3.5, 0.9, "Top-K Strategies", C_OUT,
                 subtext="DP/PP/TP/CP/EP + memory breakdown")
draw_rounded_box(ax, (5, 0.5), 3.5, 0.9, "Allocation Mode", C_OUT,
                 subtext="megatron:(attn:d1p8t8|ffn:...)")
draw_rounded_box(ax, (9, 0.5), 3.5, 0.9, "JSON / CLI Output", C_OUT,
                 subtext="--json, --top N, --find-min-nodes")
ax.text(6.5, 0.15, "OUTPUT", ha="center", fontsize=9, fontweight="bold", color="#666")

draw_arrow(ax, (5.0, 2.2), (2.75, 1.45), color=C_CORE,
           connectionstyle="arc3,rad=0.1")
draw_arrow(ax, (6.0, 2.2), (6.75, 1.45), color=C_CORE)
draw_arrow(ax, (7.0, 2.2), (10.75, 1.45), color=C_CORE,
           connectionstyle="arc3,rad=-0.1")

plt.tight_layout()
fig.savefig("/tmp/AutoParallel/docs/images/architecture.png", dpi=180,
            bbox_inches="tight", facecolor="white")
fig.savefig("/tmp/AutoParallel/docs/images/architecture.svg",
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("architecture done")
