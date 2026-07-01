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

def pill(x, y, w, h, label, detail, color, emphasize=False, fontsize=12.5):
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
ax.text(0.5, 7.15, "BUILDING RAG FROM SCRATCH  ·  BUG POSTMORTEM #02", ha="left", va="center",
        fontsize=10.5, fontweight="bold", color=CYAN, fontfamily="monospace")
ax.text(6.5, 6.55, "The Eval That Lied About Citations",
        ha="center", va="center", fontsize=20, fontweight="bold", color=WHITE)
ax.text(6.5, 6.1, "bot cited correctly · eval scored 0 · the bug was in the metric, not the model",
        ha="center", va="center", fontsize=11.5, color=GRAY, fontfamily="monospace")

ax.plot([0.5, 12.5], [5.6, 5.6], color=LINE, linewidth=1.2)

# ---------------- Row 1: The bug ----------------
ax.text(0.5, 5.25, "THE BUG  ·  RIGID REGEX, WRONG FORMAT ASSUMED", ha="left", va="center",
        fontsize=11.5, fontweight="bold", color=ROSE)

y1, h1, w1, gap1 = 3.55, 1.45, 3.6, 0.55
n1 = 3
x0_1 = (13 - (n1 * w1 + (n1 - 1) * gap1)) / 2

row1 = [
    ("BOT CITES", 'doc.pdf (p. 1)\n— valid, natural format', ROSE, False),
    ("STRICT REGEX", r'expected (doc.pdf p. 1)' + '\nboth inside parens', ROSE, False),
    ("✗ SCORE: 0.00", "citation_match fails\nbot did nothing wrong", ROSE, True),
]
xs1 = []
for i, (lab, det, col, emph) in enumerate(row1):
    x = x0_1 + i * (w1 + gap1)
    xs1.append(x)
    pill(x, y1, w1, h1, lab, det, col, emphasize=emph, fontsize=13)
    if i < n1 - 1:
        arrow(x + w1, y1 + h1 / 2, x + w1 + gap1, y1 + h1 / 2)

# ---------------- Row 2: The fix ----------------
ax.text(0.5, 2.35, "THE FIX  ·  INSPECT REAL OUTPUT, LOOSEN THE MATCHER", ha="left", va="center",
        fontsize=11.5, fontweight="bold", color=GREEN)

y2, h2 = 0.65, 1.45
x0_2 = x0_1
row2 = [
    ("INSPECT OUTPUT", "look at what the model\nactually emits first", AMBER, False),
    ("LOOSE REGEX", "accept all common\ncitation variants", PURPLE, False),
    ("✓ SCORE: 1.00", "citation_match: 0.33→1.00\nreal failures visible now", GREEN, True),
]
xs2 = []
for i, (lab, det, col, emph) in enumerate(row2):
    x = x0_2 + i * (w1 + gap1)
    xs2.append(x)
    pill(x, y2, w1, h2, lab, det, col, emphasize=emph, fontsize=13)
    if i < n1 - 1:
        arrow(x + w1, y2 + h2 / 2, x + w1 + gap1, y2 + h2 / 2)

# Connector: ✗ SCORE (before) -> ✓ SCORE (after)
arrow(xs1[2] + w1 / 2, y1, xs2[2] + w1 / 2, y2 + h2, color=WHITE, lw=1.8, ls=(0, (4, 3)))
mid_x = xs1[2] + w1 / 2
mid_y = (y1 + (y2 + h2)) / 2
ax.text(mid_x + 0.15, mid_y, "  same bot output,\n  fixed metric", ha="left", va="center",
        fontsize=8.5, color=WHITE, fontfamily="monospace")

# ---------------- Footer ----------------
ax.text(6.5, 0.32,
        "eval bugs and real bugs look identical on the dashboard — diagnose before you fix",
        ha="center", va="center", fontsize=10.5, color=GRAY,
        fontfamily="monospace", style="italic")
ax.text(12.5, 0.32, "BUG #02", ha="right", va="center",
        fontsize=10, color=GRAY, fontfamily="monospace")

plt.tight_layout()
plt.savefig("/home/user/RAG/issue_02_citation_regex_dark.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("saved")
