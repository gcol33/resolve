"""Build the RESOLVE hex logo: one vegetation plot of the example survey seen
at an angle, its plants drawn on a quadrat grid that bends away below them.

The plants are the species of the most species-rich plot in
``r/inst/extdata`` (ties broken by plot id), each species given a number of
plants in proportion to its cover and coloured by growth form. The grid's
horizontal lines drift towards the viewer by one cell per cycle, so the loop
closes on its own first frame; everything else is still.

Writes ``r/man/figures/logo.svg``.
Run from anywhere: ``python tools/figures/logo.py``.
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict

from palette import (
    BLUE,
    BORDER,
    EXAMPLE_DATA,
    FORM_COLOUR,
    GREY,
    GROWTH_FORM,
    PAPER,
    R_FIGURES,
    SERIF,
    fmt,
)

WIDTH, HEIGHT = 520, 600
CX, CY = 260.0, 300.0

# Plane (unbent) coordinates: x across the hex from its centre, v down from
# the top of the viewBox. The plane is flat down to BEND_START and then turns
# away from the viewer over BEND_LENGTH plane units, like the surface of a
# globe below its equator.
BEND_START = 150.0
BEND_LENGTH = 330.0
TURN = 1.25
LIFT = 55.0
NARROW = 0.16
PLANE_END = BEND_START + BEND_LENGTH

CELL = 46.0
CYCLE_SECONDS = 8
FRAMES = 16
LINE_POINTS = 25
N_PLANTS = 16
PLANT_RADIUS = 12.0
POOL_POINT = (CX, 384.0)


def bend(v: float) -> float:
    """Bend parameter t in [0, 1] of plane row v."""
    return min(max((v - BEND_START) / BEND_LENGTH, 0.0), 1.0)


def project(x: float, v: float) -> tuple[float, float]:
    t = bend(v)
    if t == 0.0:
        y = v
    else:
        y = BEND_START + BEND_LENGTH * math.sin(t * TURN) / TURN
    u = x / (WIDTH / 2)
    scale = 1.0 - NARROW * (1.0 - math.cos(t * TURN))
    return CX + x * scale, y - LIFT * u * u * t * t


def foreshortening(v: float) -> tuple[float, float]:
    """Horizontal and vertical scale of a small circle lying on the plane."""
    t = bend(v)
    return 1.0 - NARROW * (1.0 - math.cos(t * TURN)), math.cos(t * TURN)


def polyline(points: list[tuple[float, float]]) -> str:
    head, *rest = points
    return f"M{fmt(head[0])} {fmt(head[1])}" + "".join(
        f"L{fmt(x)} {fmt(y)}" for x, y in rest
    )


def hex_points(r: float) -> str:
    pts = []
    for k in range(6):
        a = math.radians(-90 + 60 * k)
        pts.append(f"{fmt(CX + r * math.cos(a))},{fmt(CY + r * math.sin(a))}")
    return " ".join(pts)


def richest_plot() -> list[tuple[str, str, float]]:
    rows = defaultdict(list)
    with open(EXAMPLE_DATA / "species.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["plot_id"]].append((r["species"], r["family"], float(r["cover"])))
    plot = min(rows, key=lambda p: (-len(rows[p]), p))
    return sorted(rows[plot], key=lambda s: -s[2])


def plant_counts(species: list[tuple[str, str, float]]) -> list[tuple[str, int]]:
    total = sum(c for _, _, c in species)
    counts = [max(1, round(N_PLANTS * c / total)) for _, _, c in species]
    return [(GROWTH_FORM[f], n) for (_, f, _), n in zip(species, counts)]


def place_plants(counts: list[tuple[str, int]]) -> list[tuple[float, float, str]]:
    rng = random.Random(20261031)
    placed: list[tuple[float, float, str]] = []
    for form, n in counts:
        for _ in range(n):
            for _attempt in range(4000):
                x = rng.uniform(-205.0, 205.0)
                v = rng.uniform(70.0, 330.0)
                sx, sy = project(x, v)
                if not inside_hex(sx, sy, 250.0):
                    continue
                if all(math.hypot(x - px, v - pv) > 3.1 * PLANT_RADIUS for px, pv, _ in placed):
                    placed.append((x, v, form))
                    break
            else:
                raise RuntimeError("could not place every plant; lower N_PLANTS")
    return placed


def inside_hex(x: float, y: float, r: float) -> bool:
    dx, dy = abs(x - CX), abs(y - CY)
    half = r * math.sqrt(3) / 2
    return dx <= half and dy <= r - dx / math.sqrt(3)


def vertical_lines() -> list[str]:
    paths = []
    j_max = int(WIDTH / CELL) + 1
    for j in range(-j_max, j_max + 1):
        x = j * CELL
        pts = [project(x, PLANE_END * k / (LINE_POINTS - 1)) for k in range(LINE_POINTS)]
        paths.append(polyline(pts))
    return paths


def horizontal_line(v: float) -> str:
    v = min(v, PLANE_END)
    half = WIDTH * 0.62
    xs = [-half + 2 * half * k / (LINE_POINTS - 1) for k in range(LINE_POINTS)]
    return polyline([project(x, v) for x in xs])


def horizontal_lines(offset: float) -> list[str]:
    n = int(PLANE_END / CELL) + 2
    return [horizontal_line(-CELL + k * CELL + offset) for k in range(n)]


def fade_mask() -> str:
    """Opacity strips that fade the plane out over the far part of its bend."""
    strips = []
    steps = 9
    t0 = 0.55
    half = WIDTH * 0.7
    for s in range(steps + 1):
        ta = t0 + (1 - t0) * s / (steps + 1)
        tb = t0 + (1 - t0) * (s + 1) / (steps + 1)
        va, vb = BEND_START + ta * BEND_LENGTH, BEND_START + tb * BEND_LENGTH
        xs = [-half + 2 * half * k / 20 for k in range(21)]
        top = [project(x, va) for x in xs]
        bottom = [project(x, vb) for x in reversed(xs)]
        opacity = 1 - (s + 1) / (steps + 1)
        strips.append(
            f'<path d="{polyline(top + bottom)}Z" fill="#fff" fill-opacity="{opacity:.1f}"/>'
        )
    full_end = BEND_START + t0 * BEND_LENGTH
    xs = [-half + 2 * half * k / 20 for k in range(21)]
    lower = [project(x, full_end) for x in reversed(xs)]
    top_rect = [(CX - half, -10.0), (CX + half, -10.0)] + lower
    return f'<path d="{polyline(top_rect)}Z" fill="#fff"/>' + "".join(strips)


def pooling_lines(plants: list[tuple[float, float, str]]) -> str:
    """Lines from every plant to the one point the plot is pooled into."""
    px, py = POOL_POINT
    out = []
    for x, v, form in plants:
        sx, sy = project(x, v)
        out.append(
            f'<path d="M{fmt(sx)} {fmt(sy)}L{fmt(px)} {fmt(py)}" stroke="{FORM_COLOUR[form]}"/>'
        )
    return (
        f'<g fill="none" stroke-width="2.5" stroke-opacity="0.55">{"".join(out)}</g>'
        f'<circle cx="{fmt(px)}" cy="{fmt(py)}" r="10" fill="{BLUE}" stroke="{PAPER}" stroke-width="4"/>'
    )


def plant_marks(plants: list[tuple[float, float, str]]) -> str:
    out = []
    for x, v, form in sorted(plants, key=lambda p: p[1]):
        sx, sy = project(x, v)
        fx, fy = foreshortening(v)
        colour = FORM_COLOUR[form]
        out.append(
            f'<ellipse cx="{fmt(sx)}" cy="{fmt(sy)}" rx="{fmt(PLANT_RADIUS * fx)}" '
            f'ry="{fmt(PLANT_RADIUS * fy)}" fill="{colour}" stroke="{PAPER}" stroke-width="3"/>'
        )
    return "".join(out)


def build() -> str:
    plants = place_plants(plant_counts(richest_plot()))

    frames = [horizontal_lines(CELL * f / FRAMES) for f in range(FRAMES + 1)]
    moving = []
    for k in range(len(frames[0])):
        values = ";".join(frame[k] for frame in frames)
        moving.append(
            f'<path d="{frames[0][k]}"><animate attributeName="d" values="{values}" '
            f'dur="{CYCLE_SECONDS}s" calcMode="linear" repeatCount="indefinite"/></path>'
        )
    still = "".join(f'<path d="{d}"/>' for d in frames[0])
    verticals = "".join(f'<path d="{d}"/>' for d in vertical_lines())

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="RESOLVE logo">
<title>RESOLVE logo</title>
<style>
.still {{ display: none; }}
@media (prefers-reduced-motion: reduce) {{ .moving {{ display: none; }} .still {{ display: inline; }} }}
</style>
<defs>
<clipPath id="hex"><polygon points="{hex_points(288)}"/></clipPath>
<mask id="fade" maskUnits="userSpaceOnUse" x="0" y="0" width="{WIDTH}" height="{HEIGHT}">{fade_mask()}</mask>
</defs>
<polygon points="{hex_points(288)}" fill="{PAPER}"/>
<g clip-path="url(#hex)">
<g mask="url(#fade)">
<g fill="none" stroke="{GREY}" stroke-width="2" stroke-opacity="0.45" stroke-linecap="round">
{verticals}
<g class="moving">{"".join(moving)}</g>
<g class="still">{still}</g>
</g>
</g>
{pooling_lines(plants)}
{plant_marks(plants)}
<text x="{fmt(CX)}" y="472" text-anchor="middle" font-family="{SERIF}" font-size="66" font-weight="bold" font-style="italic" fill="{BLUE}" stroke="{PAPER}" stroke-width="10" stroke-linejoin="round" paint-order="stroke">RESOLVE</text>
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
