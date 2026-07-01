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

def pill(x, y, w, h, label, detail, color, emphasize=False, fontsize=11.5):
    rounding = 0.25
    fill = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding:.2f}",
        linewidth=0, facecolor=color, alpha=0.22 if emphasize else 0.10,
    )
    ax.add_patch(fill)
    outline = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding:.2f}",
        linewidth=1.8, edgecolor=color, facecolor="none",
    )
    ax.add_patch(outline)
    ax.text(x + w / 2, y + h * 0.68, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.32, detail, ha="center", va="center",
            fontsize=8.2, color=GRAY, fontfamily="monospace")

# ---------------- Header ----------------
ax.text(0.5, 7.15, "BUILDING RAG FROM SCRATCH  ·  COMMON PITFALLS", ha="left", va="center",
        fontsize=10.5, fontweight="bold", color=CYAN, fontfamily="monospace")
ax.text(6.5, 6.6, "4 RAG Eval Mistakes Almost Everyone Makes",
        ha="center", va="center", fontsize=20, fontweight="bold", color=WHITE)
ax.text(6.5, 6.15, "low scores can mean broken pipeline — or broken measurement",
        ha="center", va="center", fontsize=12, color=GRAY, fontfamily="monospace")

ax.plot([0.5, 12.5], [5.75, 5.75], color=LINE, linewidth=1.2)

# 2x2 grid of pitfalls
pw, ph = 5.6, 2.1
col1_x = 0.6
col2_x = 6.8
row1_y = 3.35
row2_y = 0.85

cells = [
    (col1_x, row1_y, "① PROMPTS DON'T ENFORCE FORMAT",
     'wrote "(doc.pdf p.N)"  →  model emits\n"doc.pdf (p.N)"  →  regex scores 0\nsystem prompts are suggestions, not contracts', ROSE),

    (col2_x, row1_y, "② EVAL BUGS LOOK LIKE CODE BUGS",
     "citation_match: 0/3 → pipeline must be broken?\nnope. the metric was wrong, not the bot\nalways inspect actual output before fixing", AMBER),

    (col1_x, row2_y, "③ END-TO-END SCORES HIDE STAGE FAILURES",
     "rewriter answered instead of rephrasing\nretrieval still worked (lucky keywords)\neval said PASS — outcome metrics can't see it", PURPLE),

    (col2_x, row2_y, "④ KEYWORDS ≠ MEANING",
     '"no cream" ≠ "does not use cream" to a regex\nkeyword_coverage fails on valid synonyms\nuse llm_judge for semantic checks', CYAN),
]

for (x, y, label, detail, color) in cells:
    pill(x, y, pw, ph, label, detail, color, fontsize=11.5)

# ---------------- Footer ----------------
ax.text(6.5, 0.38,
        "every red metric is a question: is the pipeline wrong, or is the measurement wrong?",
        ha="center", va="center", fontsize=10.5, color=GRAY,
        fontfamily="monospace", style="italic")

plt.tight_layout()
plt.savefig("/home/user/RAG/rag_eval_pitfalls_dark.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("saved")
