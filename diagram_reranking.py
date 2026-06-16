import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 7.5), dpi=150)
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.5)
ax.axis("off")

BG = "#0B1220"
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

WHITE = "#F8FAFC"
GRAY = "#7C8AA5"
LINE = "#27324A"
CYAN = "#22D3EE"
PURPLE = "#A78BFA"
GREEN = "#34D399"
AMBER = "#FBBF24"
ROSE = "#FB7185"

def pill(x, y, w, h, label, detail, color, emphasize=False):
    rounding = h / 2
    fill = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding:.2f}",
        linewidth=0, facecolor=color, alpha=0.25 if emphasize else 0.12,
    )
    ax.add_patch(fill)
    outline = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding:.2f}",
        linewidth=1.8, edgecolor=color, facecolor="none",
    )
    ax.add_patch(outline)
    ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
            fontsize=12, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center",
            fontsize=8.5, color=GRAY, fontfamily="monospace")

def arrow(x1, y1, x2, y2, color=GRAY, lw=1.8, ls="solid", connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=color,
                         linewidth=lw, mutation_scale=15, linestyle=ls,
                         connectionstyle=connectionstyle)
    ax.add_patch(a)

# ---------------- Header ----------------
ax.text(0.5, 7.15, "BUILDING RAG FROM SCRATCH  ·  PART 3", ha="left", va="center",
        fontsize=10.5, fontweight="bold", color=CYAN, fontfamily="monospace")
ax.text(6.5, 6.6, "Re-ranking — Bi-encoder for Recall, Cross-encoder for Precision",
        ha="center", va="center", fontsize=18.5, fontweight="bold", color=WHITE)
ax.text(6.5, 6.15, "Retrieve wide (k=20)  →  Re-score all pairs  →  Keep best 4",
        ha="center", va="center", fontsize=12, color=GRAY, fontfamily="monospace")

ax.plot([0.5, 12.5], [5.72, 5.72], color=LINE, linewidth=1.2)

# ---------------- Row 1: Before ----------------
ax.text(0.5, 5.42, "BEFORE  ·  SINGLE-STAGE RETRIEVAL", ha="left", va="center",
        fontsize=11, fontweight="bold", color=ROSE)

y1, h1, w1, gap1 = 3.95, 1.3, 2.7, 0.55
n1 = 4
x0 = (13 - (n1 * w1 + (n1 - 1) * gap1)) / 2

row1 = [
    ("QUESTION",    "user query",          ROSE, False),
    ("BI-ENCODER",  "FAISS  ·  top-4",    ROSE, False),
    ("TOP-4 CHUNKS","rough ranking",       ROSE, False),
    ("✗ NOISE",     "right chunk @ rank 7",ROSE, True),
]
xs1 = []
for i, (lab, det, col, emph) in enumerate(row1):
    x = x0 + i * (w1 + gap1)
    xs1.append(x)
    pill(x, y1, w1, h1, lab, det, col, emphasize=emph)
    if i < n1 - 1:
        arrow(x + w1, y1 + h1 / 2, x + w1 + gap1, y1 + h1 / 2)

# ---------------- Row 2: After ----------------
ax.text(0.5, 2.50, "AFTER  ·  TWO-STAGE RETRIEVAL", ha="left", va="center",
        fontsize=11, fontweight="bold", color=GREEN)

y2, h2 = 0.95, 1.3
row2 = [
    ("QUESTION",       "user query",             CYAN,   False),
    ("BI-ENCODER",     "FAISS  ·  top-20",       PURPLE, False),
    ("CROSS-ENCODER",  "rerank  →  top-4",       AMBER,  False),
    ("✓ PRECISE",      "BAAI/bge-reranker-base", GREEN,  True),
]
xs2 = []
for i, (lab, det, col, emph) in enumerate(row2):
    x = x0 + i * (w1 + gap1)
    xs2.append(x)
    pill(x, y2, w1, h2, lab, det, col, emphasize=emph)
    if i < n1 - 1:
        arrow(x + w1, y2 + h2 / 2, x + w1 + gap1, y2 + h2 / 2)

# Connector: TOP-4 CHUNKS (row1, pill 2) → CROSS-ENCODER (row2, pill 2)
# Same x position — clean vertical dashed arrow
cx = xs1[2] + w1 / 2
arrow(cx, y1, cx, y2 + h2, color=WHITE, lw=1.8, ls=(0, (4, 3)))
ax.text(cx + 0.12, (y1 + y2 + h2) / 2, "  insert re-ranker\n  between these",
        ha="left", va="center", fontsize=8.5, color=WHITE, fontfamily="monospace")

# ---------------- Footer ----------------
ax.text(6.5, 0.42,
        "bi-encoder for recall (cast wide)  ·  cross-encoder for precision (pick the best)",
        ha="center", va="center", fontsize=10.5, color=GRAY,
        fontfamily="monospace", style="italic")
ax.text(12.5, 0.42, "03 / 06", ha="right", va="center",
        fontsize=10, color=GRAY, fontfamily="monospace")

plt.tight_layout()
plt.savefig("/home/user/RAG/reranking_pipeline_dark.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("saved")
