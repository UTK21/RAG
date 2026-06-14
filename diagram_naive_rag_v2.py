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
PINK = "#F472B6"

def pill(x, y, w, h, label, detail, color):
    rounding = h / 2
    fill = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding:.2f}",
        linewidth=0, facecolor=color, alpha=0.12,
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

def arrow(x1, y1, x2, y2, color=GRAY, lw=1.8, connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=color,
                         linewidth=lw, mutation_scale=15,
                         connectionstyle=connectionstyle)
    ax.add_patch(a)

# ---------------- Header ----------------
ax.text(0.5, 7.15, "BUILDING RAG FROM SCRATCH  ·  PART 1", ha="left", va="center",
        fontsize=10.5, fontweight="bold", color=CYAN, fontfamily="monospace")
ax.text(6.5, 6.6, "Naive RAG — the Baseline Pipeline", ha="center", va="center",
        fontsize=22, fontweight="bold", color=WHITE)
ax.text(6.5, 6.15, "PDF  →  Chunks  →  Embeddings  →  FAISS  →  LLM  →  Answer",
        ha="center", va="center", fontsize=12, color=GRAY, fontfamily="monospace")

ax.plot([0.5, 12.5], [5.72, 5.72], color=LINE, linewidth=1.2)

# ---------------- Row 1: Indexing ----------------
ax.text(0.5, 5.42, "ONE-TIME  ·  BUILD THE INDEX", ha="left", va="center",
        fontsize=11, fontweight="bold", color=CYAN)

y1, h1, w1, gap1 = 3.95, 1.3, 2.7, 0.55
n1 = 4
x0_1 = (13 - (n1 * w1 + (n1 - 1) * gap1)) / 2
row1 = [
    ("PDF", "your document", CYAN),
    ("CHUNKS", "800w / 150 overlap", PURPLE),
    ("EMBEDDINGS", "all-MiniLM-L6-v2", GREEN),
    ("FAISS INDEX", "vector store (IP)", AMBER),
]
xs1 = []
for i, (lab, det, col) in enumerate(row1):
    x = x0_1 + i * (w1 + gap1)
    xs1.append(x)
    pill(x, y1, w1, h1, lab, det, col)
    if i < n1 - 1:
        arrow(x + w1, y1 + h1 / 2, x + w1 + gap1, y1 + h1 / 2)

# ---------------- Row 2: Per question ----------------
ax.text(0.5, 2.50, "PER QUESTION  ·  RETRIEVE & ANSWER", ha="left", va="center",
        fontsize=11, fontweight="bold", color=PURPLE)

y2, h2, w2, gap2 = 0.95, 1.3, 2.15, 0.35
n2 = 5
x0_2 = (13 - (n2 * w2 + (n2 - 1) * gap2)) / 2
row2 = [
    ("QUESTION", "user input", PINK),
    ("EMBED", "same bi-encoder", CYAN),
    ("SEARCH", "top-k nearest", AMBER),
    ("LLM", "+ grounding prompt", PURPLE),
    ("ANSWER", "w/ page citation", GREEN),
]
xs2 = []
for i, (lab, det, col) in enumerate(row2):
    x = x0_2 + i * (w2 + gap2)
    xs2.append(x)
    pill(x, y2, w2, h2, lab, det, col)
    if i < n2 - 1:
        arrow(x + w2, y2 + h2 / 2, x + w2 + gap2, y2 + h2 / 2)

# Connector: FAISS INDEX -> SEARCH (same accent color, ties the two rows together)
arrow(xs1[3] + w1 / 2, y1, xs2[2] + w2 / 2, y2 + h2, color=AMBER, lw=2.0)

# ---------------- Footer ----------------
ax.text(6.5, 0.42, '"Use ONLY this context. Cite page numbers."',
        ha="center", va="center", fontsize=10.5, color=GRAY,
        fontfamily="monospace", style="italic")
ax.text(12.5, 0.42, "01 / 06", ha="right", va="center",
        fontsize=10, color=GRAY, fontfamily="monospace")

plt.tight_layout()
plt.savefig("/home/user/RAG/naive_rag_pipeline_dark.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("saved")
