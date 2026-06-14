import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

fig, ax = plt.subplots(figsize=(13, 6.2), dpi=150)
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.2)
ax.axis("off")

# Palette
BLUE_FILL, BLUE_EDGE = "#DCEBFE", "#2563EB"
GREEN_FILL, GREEN_EDGE = "#DCFCE7", "#16A34A"
PURPLE_FILL, PURPLE_EDGE = "#F3E8FF", "#9333EA"
GRAY = "#6B7280"
DARK = "#111827"

def box(x, y, w, h, text, fill, edge, fontsize=10.5, fontweight="bold"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=DARK, linespacing=1.3)

def arrow(x1, y1, x2, y2, color=GRAY, style="-|>", lw=2, connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                         linewidth=lw, mutation_scale=18,
                         connectionstyle=connectionstyle)
    ax.add_patch(a)

# Title
ax.text(6.5, 5.95, "Naive RAG — the baseline pipeline", ha="center", va="center",
        fontsize=18, fontweight="bold", color=DARK)
ax.text(6.5, 5.55, "PDF  →  chunks  →  embeddings  →  FAISS  →  LLM  →  grounded answer",
        ha="center", va="center", fontsize=11, color=GRAY)

# ---- Row 1: Indexing (one-time) ----
ax.text(0.3, 4.75, "ONE-TIME · BUILD THE INDEX", fontsize=10, fontweight="bold", color=BLUE_EDGE)

y1 = 3.55
h1 = 1.0
w1 = 2.55
xs1 = [0.3, 3.25, 6.2, 9.15]

box(xs1[0], y1, w1, h1, "PDF\n(your document)", BLUE_FILL, BLUE_EDGE)
box(xs1[1], y1, w1, h1, "Chunks\n~800 words,\n150-word overlap", BLUE_FILL, BLUE_EDGE)
box(xs1[2], y1, w1, h1, "Embeddings\nall-MiniLM-L6-v2\n(vector per chunk)", BLUE_FILL, BLUE_EDGE)
box(xs1[3], y1, w1, h1, "FAISS Index\nvector store", BLUE_FILL, BLUE_EDGE)

for i in range(3):
    arrow(xs1[i] + w1, y1 + h1 / 2, xs1[i + 1], y1 + h1 / 2)

# ---- Row 2: Per question ----
ax.text(0.3, 2.3, "PER QUESTION · RETRIEVE & ANSWER", fontsize=10, fontweight="bold", color=GREEN_EDGE)

y2 = 1.1
h2 = 1.0
w2 = 1.95
xs2 = [0.3, 2.6, 4.9, 7.2, 9.5]

box(xs2[0], y2, w2, h2, "User\nquestion", GREEN_FILL, GREEN_EDGE)
box(xs2[1], y2, w2, h2, "Embed\nquestion", GREEN_FILL, GREEN_EDGE)
box(xs2[2], y2, w2, h2, "Search FAISS\ntop-k nearest", GREEN_FILL, GREEN_EDGE)
box(xs2[3], y2, w2, h2, "LLM\n+ grounding\nprompt", PURPLE_FILL, PURPLE_EDGE, fontsize=10)
box(xs2[4], y2, w2 + 0.5, h2, "Grounded\nanswer\n(w/ page cite)", GREEN_FILL, GREEN_EDGE)

for i in range(3):
    arrow(xs2[i] + w2, y2 + h2 / 2, xs2[i + 1], y2 + h2 / 2)
arrow(xs2[3] + w2, y2 + h2 / 2, xs2[4], y2 + h2 / 2)

# Link FAISS index (row1) down to Search FAISS (row2)
arrow(xs1[3] + w1 / 2, y1, xs2[2] + w2 / 2, y2 + h2,
      color=BLUE_EDGE, connectionstyle="arc3,rad=0.0")

# Grounding note under LLM box
ax.text(xs2[3] + w2 / 2, y2 - 0.42,
        "system prompt: “Use ONLY this context.\nCite page numbers.”",
        ha="center", va="center", fontsize=8.5, color=GRAY, style="italic")

plt.tight_layout()
plt.savefig("/home/user/RAG/naive_rag_pipeline.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("saved")
