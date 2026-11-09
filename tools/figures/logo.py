"""Build the RESOLVE hex logo: the species of one vegetation plot flowing into
a single point, the plot's representation, and out again into the seven
things the model reads from it (area, elevation, slope, aspect, survey year,
location, habitat).

The species are the five with the most cover in the most species-rich plot
in ``r/inst/extdata`` (ties broken by plot id), sized by cover and coloured
by growth form. Small
pulses travel along the curves from the species to the outputs, each on a
loop of one cycle, so the animation closes on its own first frame; a species
dips briefly as its pulse leaves it, an output swells as its pulse runs into
it, and the name is still.

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

N_SPECIES = 5
N_TARGETS = 7
LATENT = (CX, 230.0)
LATENT_RADIUS = 20.0
INPUT_ARC = (CX, 330.0, 235.0)  # centre x, centre y, radius of the arc the species sit on
INPUT_SPREAD = 50.0  # degrees either side of straight up
SPECIES_RADIUS = (11.0, 9.0)  # smallest radius, extra radius at the largest cover
OUTPUT_Y, OUTPUT_HALF = 338.0, 174.0
OUTPUT_SAG = 20.0
OUTPUT_RADIUS = 17.0
OUTPUT_STROKE = 7.0
EDGE_WIDTH = 5.0
EDGE_OPACITY = 0.45
CYCLE_SECONDS = 6
PULSE_RADIUS = 6.5
ABSORB_SCALE = 1.14  # an output's size at the peak of its swell as a pulse arrives
EMIT_SCALE = 0.86  # a species' size at the low of its dip as a pulse leaves
SWELL_PEAK, SWELL_END = 0.04, 0.14  # fractions of the cycle
NAME_Y, NAME_SIZE = 442.0, 80.0


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
    return sorted(rows[plot], key=lambda s: -s[1])[:N_SPECIES]


def input_nodes(species: list[tuple[str, float]]) -> list[tuple[float, float, float, str]]:
    """Species on an arc across the top, largest cover in the middle."""
    n = len(species)
    order = sorted(range(n), key=lambda i: abs(i - (n - 1) / 2))
    top = max(c for _, c in species)
    cx, cy, r = INPUT_ARC
    nodes = []
    for slot, (form, cover) in zip(order, species):
        angle = math.radians(-90 - INPUT_SPREAD + 2 * INPUT_SPREAD * slot / max(n - 1, 1))
        radius = SPECIES_RADIUS[0] + SPECIES_RADIUS[1] * math.sqrt(cover / top)
        nodes.append((cx + r * math.cos(angle), cy + r * math.sin(angle), radius, FORM_COLOUR[form]))
    return nodes


def output_nodes() -> list[tuple[float, float]]:
    nodes = []
    for k in range(N_TARGETS):
        u = -1 + 2 * k / (N_TARGETS - 1)
        nodes.append((CX + OUTPUT_HALF * u, OUTPUT_Y - OUTPUT_SAG * (1 - u * u)))
    return nodes


Point = tuple[float, float]


def curve_controls(a: Point, b: Point) -> tuple[Point, Point, Point, Point]:
    """Control points of the cubic from a to b leaving and arriving vertically."""
    (x0, y0), (x1, y1) = a, b
    k = 0.55 * (y1 - y0)
    return (x0, y0), (x0, y0 + k), (x1, y1 - k), (x1, y1)


def curve(a: Point, b: Point) -> str:
    (p0, p1, p2, p3) = curve_controls(a, b)
    return "M{} {}C{} {} {} {} {} {}".format(*(fmt(v) for q in (p0, p1, p2, p3) for v in q))


def rim_fraction(a: Point, b: Point, centre: Point, radius: float, samples: int = 400) -> float:
    """Fraction of the curve's length at which it crosses the circle round `centre`.

    For `centre` = b it is where the curve first comes within `radius` of its
    end, for `centre` = a where it first leaves `radius` of its start.
    animateMotion with keyPoints 0;1 moves at constant speed along the arc,
    so this is also the fraction of the cycle at which a pulse crosses that rim.
    """
    p0, p1, p2, p3 = curve_controls(a, b)
    pts = []
    for i in range(samples + 1):
        t = i / samples
        u = 1 - t
        pts.append(tuple(
            u**3 * p0[d] + 3 * u * u * t * p1[d] + 3 * u * t * t * p2[d] + t**3 * p3[d] for d in (0, 1)
        ))
    inside_at_start = math.dist(pts[0], centre) <= radius
    total = sum(math.dist(pts[i], pts[i + 1]) for i in range(samples))
    run = 0.0
    for i, q in enumerate(pts):
        if (math.dist(q, centre) <= radius) != inside_at_start:
            return run / total
        if i < samples:
            run += math.dist(q, pts[i + 1])
    return 1.0


def pulse_begin(offset: float) -> float:
    return -offset * CYCLE_SECONDS


def swell_delay(offset: float, fraction: float, peak_at_crossing: bool) -> float:
    """animation-delay that starts a swell when the pulse crosses a rim, or centres its peak there."""
    start = pulse_begin(offset) + fraction * CYCLE_SECONDS
    if peak_at_crossing:
        start -= SWELL_PEAK * CYCLE_SECONDS
    return start % CYCLE_SECONDS - CYCLE_SECONDS


def swell_keyframes(name: str, scale: float) -> str:
    return (
        f"@keyframes {name} {{ 0% {{ transform: scale(1); }} {SWELL_PEAK:.0%} {{ transform: scale({scale}); }} "
        f"{SWELL_END:.0%}, 100% {{ transform: scale(1); }} }}"
    )


def pulse(path_id: str, colour: str, offset: float) -> str:
    """A dot travelling the path once per cycle, starting `offset` of the way in."""
    begin = pulse_begin(offset)
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
    emit_delays = []
    for i, (x, y, r, colour) in enumerate(inputs):
        pid = f"in{i}"
        edges.append(f'<path id="{pid}" d="{curve((x, y), LATENT)}"/>')
        offset = (i * 0.37) % 1.0 * 0.5
        moving.append(pulse(pid, colour, offset))
        leave = rim_fraction((x, y), LATENT, (x, y), r)
        emit_delays.append(swell_delay(offset, leave, peak_at_crossing=True))
    absorb_delays = []
    for j, (x, y) in enumerate(outputs):
        pid = f"out{j}"
        offset = 0.5 + (j * 0.29) % 1.0 * 0.5
        edges.append(f'<path id="{pid}" d="{curve(LATENT, (x, y))}"/>')
        moving.append(pulse(pid, BLUE, offset))
        arrive = rim_fraction(LATENT, (x, y), (x, y), OUTPUT_RADIUS)
        absorb_delays.append(swell_delay(offset, arrive, peak_at_crossing=False))

    species_marks = "".join(
        f'<circle class="emit" style="animation-delay:{d:.2f}s" cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(r)}" '
        f'fill="{c}" stroke="{SKY}" stroke-width="4"/>'
        for (x, y, r, c), d in zip(inputs, emit_delays)
    )
    target_marks = "".join(
        f'<circle class="absorb" style="animation-delay:{d:.2f}s" cx="{fmt(x)}" cy="{fmt(y)}" '
        f'r="{fmt(OUTPUT_RADIUS)}" fill="{SKY}" stroke="{BLUE}" stroke-width="{fmt(OUTPUT_STROKE)}"/>'
        for (x, y), d in zip(outputs, absorb_delays)
    )
    lx, ly = LATENT

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="RESOLVE logo">
<title>RESOLVE logo</title>
<style>
.emit, .absorb {{ transform-box: fill-box; transform-origin: center; animation: {CYCLE_SECONDS}s ease-out infinite; }}
.emit {{ animation-name: emit; }}
.absorb {{ animation-name: absorb; }}
{swell_keyframes("emit", EMIT_SCALE)}
{swell_keyframes("absorb", ABSORB_SCALE)}
@media (prefers-reduced-motion: reduce) {{ .moving {{ display: none; }} .emit, .absorb {{ animation: none; }} }}
</style>
<defs>
<clipPath id="hex"><polygon points="{hex_points(288)}"/></clipPath>
</defs>
<polygon points="{hex_points(288)}" fill="{SKY}"/>
<g clip-path="url(#hex)">
<g fill="none" stroke="{EDGE}" stroke-width="{fmt(EDGE_WIDTH)}" stroke-opacity="{EDGE_OPACITY}" stroke-linecap="round">{"".join(edges)}</g>
<g class="moving">{"".join(moving)}</g>
{species_marks}
<circle cx="{fmt(lx)}" cy="{fmt(ly)}" r="{fmt(LATENT_RADIUS)}" fill="{BLUE}" stroke="{SKY}" stroke-width="4"/>
{target_marks}
<text x="{fmt(CX)}" y="{fmt(NAME_Y)}" text-anchor="middle" font-family="{SERIF}" font-size="{fmt(NAME_SIZE)}" font-weight="bold" font-style="italic" fill="{BLUE}" stroke="{SKY}" stroke-width="10" stroke-linejoin="round" paint-order="stroke">RESOLVE</text>
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
