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
ax.text(0.5, 7.15, "BUILDING RAG FROM SCRATCH  ·  BUG POSTMORTEM #01", ha="left", va="center",
        fontsize=10.5, fontweight="bold", color=CYAN, fontfamily="monospace")
ax.text(6.5, 6.6, "The Rewriter That Answered Instead of Rephrasing",
        ha="center", va="center", fontsize=17.5, fontweight="bold", color=WHITE)
ax.text(6.5, 6.15, 'small LLM ignored "do not answer"  →  bug hid behind a correct final answer',
        ha="center", va="center", fontsize=11.5, color=GRAY, fontfamily="monospace")

ax.plot([0.5, 12.5], [5.72, 5.72], color=LINE, linewidth=1.2)

# ---------------- Row 1: The bug ----------------
ax.text(0.5, 5.42, "THE BUG  ·  HIDDEN BEHIND A CORRECT ANSWER", ha="left", va="center",
        fontsize=11, fontweight="bold", color=ROSE)

y1, h1, w1, gap1 = 3.95, 1.3, 2.7, 0.55
n1 = 4
x0_1 = (13 - (n1 * w1 + (n1 - 1) * gap1)) / 2

row1 = [
    ("PROMPT SAYS",     '"do NOT answer"',             ROSE, False),
    ("SMALL LLM",       "answers anyway",               ROSE, False),
    ("RETRIEVAL",       "right chunks · by luck",       ROSE, False),
    ("✗ EVAL: PASS", "outcome metrics never noticed", ROSE, True),
]
xs1 = []
for i, (lab, det, col, emph) in enumerate(row1):
    x = x0_1 + i * (w1 + gap1)
    xs1.append(x)
    pill(x, y1, w1, h1, lab, det, col, emphasize=emph)
    if i < n1 - 1:
        arrow(x + w1, y1 + h1 / 2, x + w1 + gap1, y1 + h1 / 2)

# ---------------- Row 2: The fix ----------------
ax.text(0.5, 2.50, "THE FIX  ·  FEW-SHOT + VALIDATOR + A NEW METRIC", ha="left", va="center",
        fontsize=11, fontweight="bold", color=GREEN)

y2, h2, w2, gap2 = 0.95, 1.3, 2.15, 0.35
n2 = 5
x0_2 = (13 - (n2 * w2 + (n2 - 1) * gap2)) / 2
row2 = [
    ("FEW-SHOT PROMPT",  "show, don't tell",             AMBER,  False),
    ("OUTPUT VALIDATOR", "~5 lines · falls back safely", PURPLE, False),
    ("NEW METRIC",       "rewrite_quality",               CYAN,   False),
    ("RE-RUN EVAL",      "0.67  →  1.00",                 AMBER,  False),
    ("✓ EVAL: RED→GREEN","regression caught, not shipped",GREEN,  True),
]
xs2 = []
for i, (lab, det, col, emph) in enumerate(row2):
    x = x0_2 + i * (w2 + gap2)
    xs2.append(x)
    pill(x, y2, w2, h2, lab, det, col, emphasize=emph)
    if i < n2 - 1:
        arrow(x + w2, y2 + h2 / 2, x + w2 + gap2, y2 + h2 / 2)

# Connector: ✗ EVAL: PASS (before) -> NEW METRIC (after) — the missing per-stage check
arrow(xs1[3] + w1 / 2, y1, xs2[2] + w2 / 2, y2 + h2, color=WHITE, lw=1.8, ls=(0, (4, 3)))
mid_x = (xs1[3] + w1 / 2 + xs2[2] + w2 / 2) / 2
mid_y = (y1 + (y2 + h2)) / 2
ax.text(mid_x + 0.1, mid_y, "  the missing\n  per-stage check", ha="left", va="center",
        fontsize=8.5, color=WHITE, fontfamily="monospace")

# ---------------- Footer ----------------
ax.text(6.5, 0.42,
        'an eval harness only catches what it explicitly measures — new stage, new metric',
        ha="center", va="center", fontsize=10.5, color=GRAY,
        fontfamily="monospace", style="italic")
ax.text(12.5, 0.42, "BUG #01", ha="right", va="center",
        fontsize=10, color=GRAY, fontfamily="monospace")

plt.tight_layout()
plt.savefig("/home/user/RAG/issue_01_pipeline_dark.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("saved")
