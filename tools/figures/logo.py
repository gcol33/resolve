"""Build the RESOLVE hex logo: the species of one vegetation plot flowing into
a single point, the plot's representation, and out again into the seven
things the model reads from it (area, elevation, slope, aspect, survey year,
location, habitat).

The species are those of the most species-rich plot in ``r/inst/extdata``
(ties broken by plot id), sized by cover and coloured by growth form. Small
pulses travel along the curves from the species to the outputs, each on a
loop of one cycle, so the animation closes on its own first frame; the dots
and the name are still.

Writes ``r/man/figures/logo.svg``.
Run from anywhere: ``python tools/figures/logo.py``.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict

from palette import (
    BLUE,
    BORDER,
    EXAMPLE_DATA,
    FORM_COLOUR,
    GROWTH_FORM,
    R_FIGURES,
    SERIF,
    fmt,
)

WIDTH, HEIGHT = 520, 600
CX, CY = 260.0, 300.0

SKY = "#E4ECEF"
EDGE = "#5C6166"

N_TARGETS = 7
LATENT = (CX, 262.0)
INPUT_ARC = (CX, 345.0, 250.0)  # centre x, centre y, radius of the arc the species sit on
INPUT_SPREAD = 56.0  # degrees either side of straight up
OUTPUT_Y, OUTPUT_HALF = 374.0, 174.0
OUTPUT_SAG = 22.0
OUTPUT_RADIUS = 14.0
CYCLE_SECONDS = 6
PULSE_RADIUS = 4.5


def hex_points(r: float) -> str:
    pts = []
    for k in range(6):
        a = math.radians(-90 + 60 * k)
        pts.append(f"{fmt(CX + r * math.cos(a))},{fmt(CY + r * math.sin(a))}")
    return " ".join(pts)


def richest_plot() -> list[tuple[str, float]]:
    rows = defaultdict(list)
    with open(EXAMPLE_DATA / "species.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["plot_id"]].append((GROWTH_FORM[r["family"]], float(r["cover"])))
    plot = min(rows, key=lambda p: (-len(rows[p]), p))
    return sorted(rows[plot], key=lambda s: -s[1])


def input_nodes(species: list[tuple[str, float]]) -> list[tuple[float, float, float, str]]:
    """Species on an arc across the top, largest cover in the middle."""
    n = len(species)
    order = sorted(range(n), key=lambda i: abs(i - (n - 1) / 2))
    top = max(c for _, c in species)
    cx, cy, r = INPUT_ARC
    nodes = []
    for slot, (form, cover) in zip(order, species):
        angle = math.radians(-90 - INPUT_SPREAD + 2 * INPUT_SPREAD * slot / max(n - 1, 1))
        radius = 7.0 + 7.0 * math.sqrt(cover / top)
        nodes.append((cx + r * math.cos(angle), cy + r * math.sin(angle), radius, FORM_COLOUR[form]))
    return nodes


def output_nodes() -> list[tuple[float, float]]:
    nodes = []
    for k in range(N_TARGETS):
        u = -1 + 2 * k / (N_TARGETS - 1)
        nodes.append((CX + OUTPUT_HALF * u, OUTPUT_Y - OUTPUT_SAG * (1 - u * u)))
    return nodes


def curve(a: tuple[float, float], b: tuple[float, float]) -> str:
    """Cubic from a to b leaving and arriving vertically."""
    (x0, y0), (x1, y1) = a, b
    k = 0.55 * (y1 - y0)
    return (
        f"M{fmt(x0)} {fmt(y0)}C{fmt(x0)} {fmt(y0 + k)} "
        f"{fmt(x1)} {fmt(y1 - k)} {fmt(x1)} {fmt(y1)}"
    )


def pulse(path_id: str, colour: str, offset: float) -> str:
    """A dot travelling the path once per cycle, starting `offset` of the way in."""
    begin = -offset * CYCLE_SECONDS
    return (
        f'<circle r="{fmt(PULSE_RADIUS)}" fill="{colour}">'
        f'<animateMotion dur="{CYCLE_SECONDS}s" begin="{begin:.2f}s" repeatCount="indefinite" '
        f'calcMode="linear" keyPoints="0;1" keyTimes="0;1"><mpath href="#{path_id}"/></animateMotion>'
        "</circle>"
    )


def build() -> str:
    inputs = input_nodes(richest_plot())
    outputs = output_nodes()

    edges, moving = [], []
    for i, (x, y, _, colour) in enumerate(inputs):
        pid = f"in{i}"
        edges.append(f'<path id="{pid}" d="{curve((x, y), LATENT)}"/>')
        moving.append(pulse(pid, colour, (i * 0.37) % 1.0 * 0.5))
    for j, (x, y) in enumerate(outputs):
        pid = f"out{j}"
        edges.append(f'<path id="{pid}" d="{curve(LATENT, (x, y))}"/>')
        moving.append(pulse(pid, BLUE, 0.5 + (j * 0.29) % 1.0 * 0.5))

    species_marks = "".join(
        f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(r)}" fill="{c}" stroke="{SKY}" stroke-width="3"/>'
        for x, y, r, c in inputs
    )
    target_marks = "".join(
        f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(OUTPUT_RADIUS)}" fill="{SKY}" '
        f'stroke="{BLUE}" stroke-width="5"/>'
        for x, y in outputs
    )
    lx, ly = LATENT

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="RESOLVE logo">
<title>RESOLVE logo</title>
<style>
@media (prefers-reduced-motion: reduce) {{ .moving {{ display: none; }} }}
</style>
<defs>
<clipPath id="hex"><polygon points="{hex_points(288)}"/></clipPath>
</defs>
<polygon points="{hex_points(288)}" fill="{SKY}"/>
<g clip-path="url(#hex)">
<g fill="none" stroke="{EDGE}" stroke-width="2.5" stroke-opacity="0.35" stroke-linecap="round">{"".join(edges)}</g>
<g class="moving">{"".join(moving)}</g>
{species_marks}
<circle cx="{fmt(lx)}" cy="{fmt(ly)}" r="15" fill="{BLUE}" stroke="{SKY}" stroke-width="4"/>
{target_marks}
<text x="{fmt(CX)}" y="472" text-anchor="middle" font-family="{SERIF}" font-size="70" font-weight="bold" font-style="italic" fill="{BLUE}" stroke="{SKY}" stroke-width="10" stroke-linejoin="round" paint-order="stroke">RESOLVE</text>
</g>
<polygon points="{hex_points(291)}" fill="none" stroke="{BORDER}" stroke-width="18" stroke-linejoin="round"/>
</svg>
"""


def main() -> None:
    svg = build()
    R_FIGURES.mkdir(parents=True, exist_ok=True)
    path = R_FIGURES / "logo.svg"
    path.write_text(svg, encoding="utf-8", newline="\n")
    print(f"wrote {path} ({len(svg.encode()) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
