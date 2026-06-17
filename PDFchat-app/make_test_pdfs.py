"""Generate 2 small synthetic recipe PDFs into data/ for testing the bot.

Run:
    pip install reportlab
    python make_test_pdfs.py

The two PDFs disagree on purpose (cream vs no cream in carbonara,
mild vs spicy arrabbiata) — perfect for testing multi-doc retrieval
and the "surface disagreements" prompt rule we built into llm.py.
"""
from __future__ import annotations

import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

ITALIAN_RECIPES = [
    ("Spaghetti Carbonara — The Authentic Italian Way", """
    Carbonara is a Roman dish from the mid-20th century. The authentic
    recipe uses just five ingredients: spaghetti, guanciale (cured pork
    cheek), eggs, Pecorino Romano cheese, and freshly cracked black pepper.
    There is NO cream. Adding cream is considered a mistake and is not
    traditional. The creamy texture comes entirely from the emulsion of egg
    yolks, cheese, and pasta water.

    Method: render the guanciale slowly until crisp. Whisk egg yolks with
    grated Pecorino and pepper. Reserve a cup of pasta water. Drain the
    pasta and combine off the heat with the rendered guanciale, then add
    the egg mixture, tossing rapidly. Loosen with pasta water as needed.

    Common mistakes: scrambling the eggs by adding them while the pan is
    too hot, using bacon or pancetta instead of guanciale, or adding cream.
    """),
    ("Penne all'Arrabbiata — The Spicy Roman Pasta", """
    Arrabbiata means "angry" in Italian, a reference to the dish's heat.
    The defining ingredient is red chili pepper (peperoncino), used
    generously to produce a sauce that is genuinely spicy. The other
    components are simple: garlic, olive oil, San Marzano tomatoes, and
    fresh parsley to finish.

    Method: bloom red chili and garlic in olive oil. Add crushed tomatoes
    and simmer for 20 minutes. Toss with cooked penne. Finish with parsley.

    Pecorino Romano is sometimes added but purists argue this dulls the
    heat. The dish should leave a noticeable burn — that is the entire point.
    """),
]

AMERICAN_RECIPES = [
    ("Creamy Carbonara — A Modern American Take", """
    This American-style carbonara uses heavy cream for an extra-rich sauce.
    Ingredients: spaghetti, bacon (a common substitute for guanciale),
    eggs, Parmesan cheese, heavy cream, garlic, and black pepper.

    The cream stabilizes the sauce and makes it more forgiving for home
    cooks who worry about scrambling the eggs. It also gives the dish a
    silkier, heavier texture that pairs well with the smokier flavor of
    American-cured bacon.

    Method: cook bacon until crispy and reserve fat. Sauté garlic, add
    cream and let it reduce. Whisk eggs with Parmesan. Combine pasta,
    cream sauce, and eggs off the heat. Top with bacon and pepper.

    Note: traditionalists object to this version, but it remains popular
    in American restaurants and home kitchens.
    """),
    ("Mild Arrabbiata — A Family-Friendly Version", """
    A milder take on the classic, designed for diners who prefer less
    heat. Uses a small pinch of red pepper flakes instead of fresh chili,
    sweet bell peppers for body, and a splash of cream to round the
    tomato's acidity.

    Ingredients: penne, olive oil, garlic, red bell pepper, crushed
    tomatoes, a pinch of red pepper flakes, heavy cream, basil.

    Method: sauté garlic and bell pepper, add tomatoes and a pinch of
    pepper flakes, simmer 15 minutes, stir in cream just before serving.
    Toss with penne and finish with torn basil.

    This is not traditional. It is intentionally approachable.
    """),
]


def write_pdf(filename: str, title: str, recipes: list[tuple[str, str]]):
    path = os.path.join(DATA_DIR, filename)
    doc = SimpleDocTemplate(path, pagesize=letter, title=title)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"<b>{title}</b>", styles["Title"]), Spacer(1, 18)]
    for heading, body in recipes:
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Spacer(1, 6))
        for para in body.strip().split("\n\n"):
            story.append(Paragraph(para.strip().replace("\n", " "), styles["BodyText"]))
            story.append(Spacer(1, 8))
        story.append(Spacer(1, 16))
    doc.build(story)
    print(f"  wrote {path}")


if __name__ == "__main__":
    print("Generating test PDFs into data/ ...")
    write_pdf("italian_classics.pdf", "Italian Classics Cookbook", ITALIAN_RECIPES)
    write_pdf("american_kitchen.pdf", "The American Kitchen — Pasta Edition", AMERICAN_RECIPES)
    print("Done.")
