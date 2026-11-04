"""Colours, fonts and output paths shared by the README logo and hero figure."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R_FIGURES = REPO / "r" / "man" / "figures"
EXAMPLE_DATA = REPO / "r" / "inst" / "extdata"

BORDER = "#2B3A4A"
PAPER = "#F5F6F8"
BLUE = "#2E6F9E"
RED = "#B5342B"
GREEN = "#3F7A3D"
ORANGE = "#9E5E0F"
PURPLE = "#7A4F9E"
TEAL = "#2A7A7A"
GREY = "#5C6166"

SERIF = "Georgia, 'Times New Roman', serif"
SANS = "'Segoe UI', Helvetica, Arial, sans-serif"

# Growth form of each family in the example survey, the grouping both figures
# colour species by.
GROWTH_FORM = {
    "Fagaceae": "woody",
    "Pinaceae": "woody",
    "Sapindaceae": "woody",
    "Ericaceae": "woody",
    "Salicaceae": "woody",
    "Poaceae": "graminoid",
    "Cyperaceae": "graminoid",
    "Fabaceae": "forb",
    "Ranunculaceae": "forb",
    "Caryophyllaceae": "forb",
}
FORM_COLOUR = {"woody": GREEN, "graminoid": ORANGE, "forb": PURPLE}


def fmt(x: float) -> str:
    """Shortest fixed-point text for an SVG coordinate (one decimal)."""
    s = f"{x:.1f}"
    return s[:-2] if s.endswith(".0") else s
