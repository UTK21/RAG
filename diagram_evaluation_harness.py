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
                         linewidth=lw, mutation_scale=15, linestyle=ls)
    ax.add_patch(a)

# ---------------- Header ----------------
ax.text(0.5, 7.15, "BUILDING RAG FROM SCRATCH  ·  PART 6", ha="left", va="center",
        fontsize=10.5, fontweight="bold", color=CYAN, fontfamily="monospace")
ax.text(6.5, 6.6, "Evaluation Harness — Stop Vibe-Checking Your RAG",
        ha="center", va="center", fontsize=18.5, fontweight="bold", color=WHITE)
ax.text(6.5, 6.15, "golden dataset  →  4 metrics  →  pass/fail per pipeline stage",
        ha="center", va="center", fontsize=12, color=GRAY, fontfamily="monospace")

ax.plot([0.5, 12.5], [5.72, 5.72], color=LINE, linewidth=1.2)

# ---------------- Row 1: Before ----------------
ax.text(0.5, 5.42, "BEFORE  ·  “FEELS GOOD” EVALUATION", ha="left", va="center",
        fontsize=11, fontweight="bold", color=ROSE)

y1, h1, w1, gap1 = 3.95, 1.3, 2.7, 0.55
n1 = 4
x0_1 = (13 - (n1 * w1 + (n1 - 1) * gap1)) / 2

row1 = [
    ("SHIP A CHANGE",   "new prompt / chunk size",     ROSE, False),
    ("TRY 3 QUERIES",   "by hand, in a terminal",      ROSE, False),
    ("LOOKS RIGHT",     "no number, just a feeling",   ROSE, False),
    ("✗ REGRESSION", "silent, caught by a user",   ROSE, True),
]
xs1 = []
for i, (lab, det, col, emph) in enumerate(row1):
    x = x0_1 + i * (w1 + gap1)
    xs1.append(x)
    pill(x, y1, w1, h1, lab, det, col, emphasize=emph)
    if i < n1 - 1:
        arrow(x + w1, y1 + h1 / 2, x + w1 + gap1, y1 + h1 / 2)

# ---------------- Row 2: After ----------------
ax.text(0.5, 2.50, "THE FIX  ·  EVAL HARNESS", ha="left", va="center",
        fontsize=11, fontweight="bold", color=GREEN)

y2, h2, w2, gap2 = 0.95, 1.3, 2.15, 0.35
n2 = 5
x0_2 = (13 - (n2 * w2 + (n2 - 1) * gap2)) / 2
row2 = [
    ("GOLDEN DATASET",  "dataset.yaml · Q + A + cites", AMBER,  False),
    ("RUN PIPELINE",    "eval.py · every question",     PURPLE, False),
    ("4 METRICS",       "recall · citation · keyword",  CYAN,   False),
    ("LLM JUDGE",       "metrics.py · scored 1-5",       AMBER,  False),
    ("✓ PASS / FAIL", "diff vs. last run",           GREEN,  True),
]
xs2 = []
for i, (lab, det, col, emph) in enumerate(row2):
    x = x0_2 + i * (w2 + gap2)
    xs2.append(x)
    pill(x, y2, w2, h2, lab, det, col, emphasize=emph)
    if i < n2 - 1:
        arrow(x + w2, y2 + h2 / 2, x + w2 + gap2, y2 + h2 / 2)

# Connector: SHIP A CHANGE (before) -> GOLDEN DATASET (after) — measurement replaces guessing
arrow(xs1[0] + w1 / 2, y1, xs2[0] + w2 / 2, y2 + h2, color=WHITE, lw=1.8, ls=(0, (4, 3)))
mid_x = (xs1[0] + w1 / 2 + xs2[0] + w2 / 2) / 2
mid_y = (y1 + (y2 + h2)) / 2
ax.text(mid_x + 0.1, mid_y, "  same change,\n  now a measured A/B", ha="left", va="center",
        fontsize=8.5, color=WHITE, fontfamily="monospace")

# ---------------- Footer ----------------
ax.text(6.5, 0.42,
        'each metric isolates a different stage — recall = retrieval, citation = grounding, judge = generation',
        ha="center", va="center", fontsize=10, color=GRAY,
        fontfamily="monospace", style="italic")
ax.text(12.5, 0.42, "06 / 06", ha="right", va="center",
        fontsize=10, color=GRAY, fontfamily="monospace")

plt.tight_layout()
plt.savefig("/home/user/RAG/evaluation_harness_pipeline_dark.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("saved")
