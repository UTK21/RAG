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

def pill(x, y, w, h, label, detail, color, emphasize=False, fontsize=12):
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
            fontsize=fontsize, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center",
            fontsize=8.5, color=GRAY, fontfamily="monospace")

def arrow(x1, y1, x2, y2, color=GRAY, lw=1.8, ls="solid"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=color,
                         linewidth=lw, mutation_scale=14, linestyle=ls)
    ax.add_patch(a)

# ---------------- Header ----------------
ax.text(0.5, 7.15, "BUILDING RAG FROM SCRATCH  ·  PART 4", ha="left", va="center",
        fontsize=10.5, fontweight="bold", color=CYAN, fontfamily="monospace")
ax.text(6.5, 6.6, "Hybrid Search — Dense + Sparse, Fused", ha="center", va="center",
        fontsize=20, fontweight="bold", color=WHITE)
ax.text(6.5, 6.15, "Dense (FAISS) + Sparse (BM25)  →  RRF Fusion  →  Rerank  →  LLM",
        ha="center", va="center", fontsize=12, color=GRAY, fontfamily="monospace")

ax.plot([0.5, 12.5], [5.72, 5.72], color=LINE, linewidth=1.2)

# ---------------- Section 1: blind spots ----------------
ax.text(0.5, 5.42, "THE BLIND SPOTS  ·  NEITHER ALONE IS ENOUGH", ha="left", va="center",
        fontsize=11, fontweight="bold", color=ROSE)

bw, bh, bgap = 4.5, 1.0, 0.6
bx0 = (13 - (2 * bw + bgap)) / 2
pill(bx0, 4.3, bw, bh, "DENSE (embeddings)",
     "✗ misses exact tokens — names, IDs, codes", ROSE, fontsize=12.5)
pill(bx0 + bw + bgap, 4.3, bw, bh, "SPARSE (BM25)",
     "✗ misses synonyms & paraphrasing", ROSE, fontsize=12.5)

ax.text(6.5, 3.85, "→  run both, merge the rankings", ha="center", va="center",
        fontsize=12, fontweight="bold", color=WHITE)

# ---------------- Section 2: the fix (fork -> merge) ----------------
ax.text(0.5, 3.5, "THE FIX  ·  HYBRID RETRIEVAL + RRF FUSION", ha="left", va="center",
        fontsize=11, fontweight="bold", color=GREEN)

# Question
qx, qy, qw, qh = 0.5, 1.75, 1.7, 1.0
pill(qx, qy, qw, qh, "QUESTION", "user query", CYAN, fontsize=11.5)

# Branches
brx, brw, brh = 2.65, 2.3, 0.85
dense_y, sparse_y = 2.45, 1.15
pill(brx, dense_y, brw, brh, "DENSE", "FAISS · meaning", PURPLE, fontsize=11.5)
pill(brx, sparse_y, brw, brh, "SPARSE", "BM25 · exact words", AMBER, fontsize=11.5)

# Fusion
fx, fy, fw, fh = 5.5, 1.75, 2.3, 1.0
pill(fx, fy, fw, fh, "RRF FUSION", "rank-based · no tuning", AMBER, fontsize=11.5)

# Merged result
mx, my, mw, mh = 8.35, 1.75, 2.6, 1.0
pill(mx, my, mw, mh, "✓ MERGED RANKING", "agreement wins automatically", GREEN,
     emphasize=True, fontsize=11.5)

# Arrows: question -> branches
arrow(qx + qw, qy + qh / 2, brx, dense_y + brh / 2)
arrow(qx + qw, qy + qh / 2, brx, sparse_y + brh / 2)
# branches -> fusion
arrow(brx + brw, dense_y + brh / 2, fx, fy + fh / 2)
arrow(brx + brw, sparse_y + brh / 2, fx, fy + fh / 2)
# fusion -> merged
arrow(fx + fw, fy + fh / 2, mx, my + mh / 2)

ax.text(12.0, my + mh / 2, "→ rerank\n→ LLM", ha="left", va="center",
        fontsize=9, color=GRAY, fontfamily="monospace")

# ---------------- Footer ----------------
ax.text(6.5, 0.55,
        '"agreement between retrievers wins automatically — no score tuning needed"',
        ha="center", va="center", fontsize=10.5, color=GRAY,
        fontfamily="monospace", style="italic")
ax.text(12.5, 0.55, "04 / 06", ha="right", va="center",
        fontsize=10, color=GRAY, fontfamily="monospace")

plt.tight_layout()
plt.savefig("/home/user/RAG/hybrid_search_pipeline_dark.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("saved")
