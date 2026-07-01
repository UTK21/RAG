import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 7.5), dpi=150)
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.5)
ax.axis("off")

BG = "#0D1117"          # github-dark terminal bg
PANEL = "#161B22"       # slightly lighter panel
BORDER = "#30363D"
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

WHITE  = "#E6EDF3"
GRAY   = "#7D8590"
GREEN  = "#3FB950"
RED    = "#F85149"
YELLOW = "#D29922"
CYAN   = "#79C0FF"
PURPLE = "#BC8CFF"
DIM    = "#484F58"

def panel(x, y, w, h, title=None):
    bg = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.12",
                        linewidth=1.2, edgecolor=BORDER, facecolor=PANEL)
    ax.add_patch(bg)
    if title:
        ax.text(x + 0.22, y + h - 0.22, title, ha="left", va="top",
                fontsize=8.5, color=GRAY, fontfamily="monospace")

def t(x, y, text, color=WHITE, size=9.2, bold=False, mono=True):
    ax.text(x, y, text,
            ha="left", va="top",
            fontsize=size,
            fontweight="bold" if bold else "normal",
            color=color,
            fontfamily="monospace" if mono else "sans-serif")

# ── header bar ──────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.3, 7.0), 12.4, 0.38,
    boxstyle="round,pad=0.0,rounding_size=0.0",
    linewidth=0, facecolor="#1C2128"))
t(0.55, 7.33, "$ python eval.py", color=CYAN, size=9.5)
t(9.8,  7.33, "PDFchat-app / eval.py", color=GRAY, size=8.5)

# ── LEFT panel — BEFORE ─────────────────────────────────────────────────────
panel(0.3, 0.35, 5.9, 6.55, title="BEFORE  —  citation regex too strict")

lx = 0.52   # left margin inside panel
y  = 6.55

t(lx, y, "Running 3 eval cases...", color=GRAY);          y -= 0.38

# case 1
t(lx, y, "[1] PASS  'Does carbonara use cream?'", color=GREEN, bold=True); y -= 0.32
t(lx, y, "    + retrieval_recall   1.00", color=GREEN);   y -= 0.27
t(lx, y, "    + citation_match     1.00", color=GREEN);   y -= 0.27
t(lx, y, "      cited italian_classics.pdf p.1", color=DIM, size=8.2); y -= 0.36

# case 2
t(lx, y, "[2] FAIL  'Is cream traditional?'", color=RED, bold=True); y -= 0.32
t(lx, y, "    + retrieval_recall   1.00", color=GREEN);   y -= 0.27
t(lx, y, "    - citation_match     0.00", color=RED);     y -= 0.27
t(lx, y, "      no (doc.pdf p. N) citations found", color=RED, size=8.2); y -= 0.27
t(lx, y, "      bot said: doc.pdf (p. 1)  <-- mismatch", color=YELLOW, size=8.2); y -= 0.36

# case 3
t(lx, y, "[3] FAIL  'What about a beginner cook?'", color=RED, bold=True); y -= 0.32
t(lx, y, "    + retrieval_recall   1.00", color=GREEN);   y -= 0.27
t(lx, y, "    - citation_match     0.00", color=RED);     y -= 0.27
t(lx, y, "      no (doc.pdf p. N) citations found", color=RED, size=8.2); y -= 0.42

# aggregate
ax.plot([lx, 5.9], [y + 0.08, y + 0.08], color=BORDER, linewidth=0.8)
y -= 0.15
t(lx, y, "AGGREGATE", color=WHITE, bold=True, size=9.5);  y -= 0.32
t(lx, y, "  retrieval_recall    avg=1.00   3/3", color=GREEN); y -= 0.27
t(lx, y, "  citation_match      avg=0.33   1/3", color=RED, bold=True); y -= 0.27
t(lx, y, "  passed: 1 / 3", color=RED, bold=True, size=9.5)

# ── RIGHT panel — AFTER ──────────────────────────────────────────────────────
panel(6.8, 0.35, 5.9, 6.55, title="AFTER   —  regex loosened to match variants")

rx = 7.02
y  = 6.55

t(rx, y, "Running 3 eval cases...", color=GRAY);           y -= 0.38

# case 1
t(rx, y, "[1] PASS  'Does carbonara use cream?'", color=GREEN, bold=True); y -= 0.32
t(rx, y, "    + retrieval_recall   1.00", color=GREEN);    y -= 0.27
t(rx, y, "    + citation_match     1.00", color=GREEN);    y -= 0.27
t(rx, y, "      cited italian_classics.pdf p.1", color=DIM, size=8.2); y -= 0.36

# case 2
t(rx, y, "[2] PASS  'Is cream traditional?'", color=GREEN, bold=True); y -= 0.32
t(rx, y, "    + retrieval_recall   1.00", color=GREEN);    y -= 0.27
t(rx, y, "    + citation_match     1.00", color=GREEN);    y -= 0.27
t(rx, y, "      cited italian_classics.pdf p.1", color=DIM, size=8.2); y -= 0.36

# case 3
t(rx, y, "[3] PASS  'What about a beginner cook?'", color=GREEN, bold=True); y -= 0.32
t(rx, y, "    + retrieval_recall   1.00", color=GREEN);    y -= 0.27
t(rx, y, "    + citation_match     1.00", color=GREEN);    y -= 0.27
t(rx, y, "      cited american_kitchen.pdf p.1", color=DIM, size=8.2); y -= 0.42

# aggregate
ax.plot([rx, 12.48], [y + 0.08, y + 0.08], color=BORDER, linewidth=0.8)
y -= 0.15
t(rx, y, "AGGREGATE", color=WHITE, bold=True, size=9.5);   y -= 0.32
t(rx, y, "  retrieval_recall    avg=1.00   3/3", color=GREEN); y -= 0.27
t(rx, y, "  citation_match      avg=1.00   3/3", color=GREEN, bold=True); y -= 0.27
t(rx, y, "  passed: 3 / 3", color=GREEN, bold=True, size=9.5)

# ── arrow between panels ─────────────────────────────────────────────────────
ax.annotate("", xy=(6.75, 3.6), xytext=(6.25, 3.6),
            arrowprops=dict(arrowstyle="-|>", color=WHITE, lw=2.0))
ax.text(6.27, 3.85, "fix", ha="center", va="bottom",
        fontsize=8.5, color=GRAY, fontfamily="monospace")

# ── bottom caption ────────────────────────────────────────────────────────────
ax.text(6.5, 0.16,
        "the bot was citing correctly the whole time — the regex was the bug",
        ha="center", va="bottom", fontsize=9.5, color=GRAY,
        fontfamily="monospace", style="italic")

plt.tight_layout()
plt.savefig("/home/user/RAG/issue_02_terminal_dark.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("saved")
