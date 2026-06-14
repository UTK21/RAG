import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Portrait 4:5 — good for mobile LinkedIn feed (1080x1350 @ 108dpi)
fig, ax = plt.subplots(figsize=(10, 12.5), dpi=108)
ax.set_xlim(0, 10)
ax.set_ylim(0, 12.5)
ax.axis("off")

BG = "#0B1220"
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

WHITE = "#F8FAFC"
GRAY = "#94A3B8"
LINE = "#27324A"
CYAN = "#22D3EE"
PURPLE = "#A78BFA"
GREEN = "#34D399"
AMBER = "#FBBF24"
PINK = "#F472B6"
ROSE = "#FB7185"
PANEL = "#141C2E"

def pill(x, y, w, h, text, color, fontsize=12):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={h/2:.2f}",
        linewidth=2.2, edgecolor=color, facecolor=color, alpha=0.14,
    )
    ax.add_patch(patch)
    # redraw a solid-edge outline (alpha shouldn't affect edge)
    outline = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={h/2:.2f}",
        linewidth=2.2, edgecolor=color, facecolor="none",
    )
    ax.add_patch(outline)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color)

def badge(x, y, w, h, text, color):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={h/2:.2f}",
        linewidth=0, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=11, fontweight="bold", color=BG)

# ---------------- HEADER ----------------
ax.text(5, 12.0, "RAG IS EASY", ha="center", va="center",
        fontsize=42, fontweight="bold", color=WHITE)
ax.text(5, 10.85, "...UNTIL YOU ASK A FOLLOW-UP", ha="center", va="center",
        fontsize=23, fontweight="bold", color=ROSE)
ax.text(5, 10.0, "Building a RAG chatbot from scratch  —  Part 1",
        ha="center", va="center", fontsize=13, color=GRAY)

ax.plot([0.8, 9.2], [9.3, 9.3], color=LINE, linewidth=1.5)

# ---------------- PIPELINE ----------------
caption = " ".join(list("THE BASELINE PIPELINE"))
ax.text(5, 8.75, caption, ha="center", va="center",
        fontsize=12, fontweight="bold", color=CYAN)

pill_labels = ["PDF", "CHUNKS", "VECTORS", "FAISS", "LLM", "ANSWER"]
pill_colors = [CYAN, PURPLE, GREEN, AMBER, PINK, GREEN]
n = len(pill_labels)
w_pill, gap, h_pill = 1.42, 0.13, 1.0
total_w = n * w_pill + (n - 1) * gap
x0 = (10 - total_w) / 2
y_pill = 7.05

for i, (lab, col) in enumerate(zip(pill_labels, pill_colors)):
    x = x0 + i * (w_pill + gap)
    pill(x, y_pill, w_pill, h_pill, lab, col, fontsize=12.5)
    if i < n - 1:
        ax.text(x + w_pill + gap / 2, y_pill + h_pill / 2, "›",
                ha="center", va="center", fontsize=18, color=GRAY, fontweight="bold")

ax.plot([0.8, 9.2], [6.0, 6.0], color=LINE, linewidth=1.5)

# ---------------- HOOK / PAYOFF ----------------
ax.text(5, 5.4, "BUT...", ha="center", va="center",
        fontsize=30, fontweight="bold", color=ROSE)

panel = FancyBboxPatch(
    (0.8, 2.15), 8.4, 2.75,
    boxstyle="round,pad=0.02,rounding_size=0.18",
    linewidth=1.5, edgecolor=LINE, facecolor=PANEL,
)
ax.add_patch(panel)

badge(1.15, 4.25, 0.85, 0.5, "YOU", CYAN)
ax.text(2.2, 4.5, "“What about its limitations?”", ha="left", va="center",
        fontsize=14, color=WHITE, fontweight="bold")

badge(1.15, 3.3, 0.85, 0.5, "BOT", ROSE)
ax.text(2.2, 3.55, "returns random, unrelated chunks", ha="left", va="center",
        fontsize=13, color=GRAY)

ax.text(5, 2.6, "the embedding of “its” has zero signal about what it refers to",
        ha="center", va="center", fontsize=11.5, color=GRAY, style="italic")

ax.text(5, 1.4, "Here's the fix → Part 2", ha="center", va="center",
        fontsize=15, fontweight="bold", color=WHITE)
ax.text(5, 0.6, "Post 1 of a 6-part build-in-public series",
        ha="center", va="center", fontsize=11, color=GRAY)

plt.tight_layout()
plt.savefig("/home/user/RAG/naive_rag_hook.png", dpi=108, bbox_inches="tight",
            facecolor=BG)
print("saved")
