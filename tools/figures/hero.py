"""Build the README graphical abstract from a RESOLVE fit on the example survey.

A model is fitted on ``r/inst/extdata`` (150 plots, elevation and habitat as
targets, rank-pool encoding, a fifth of the plots held out). The figure shows
one held-out plot: its species and their cover, pooled into the plot's latent
vector, and the two readouts of that vector, the elevation (against the
observed elevation of every held-out plot) and the habitat probabilities.

The fit's numbers are kept in ``tools/figures/hero_fit.json`` so the figure
redraws without retraining; ``--refit`` retrains and rewrites it (needs
``resolve_core``). Writes ``r/man/figures/resolve-hero.svg``, dark ink on a
white background.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from palette import R_FIGURES, BLUE, EXAMPLE_DATA, GREEN, GREY, RED, SANS, fmt

FIT_CACHE = Path(__file__).with_name("hero_fit.json")
HABITATS = ["alpine", "forest", "grassland", "wetland"]

W, H = 1600, 640

# One grid for every panel: equal side margins, the species rows set the
# vertical extent, and the right column's two readouts share its width and
# span the same height as the rows.
MARGIN = 80.0
ROW_STEP = 64.0
BAR_X, BAR_MAX = 400.0, 150.0
RIGHT_X, RIGHT_W = 1140.0, 380.0

# Readout titles, then every other label (species names, habitat classes).
TITLE = 36
LABEL = 30

# Dark ink on a white page; the image carries its own background, so it reads
# the same on every site theme.
THEME = {
    "background": "#FFFFFF",
    "text": GREY,
    "muted": "#8A9097",
    "rule": "#C9CDD2",
    "plant": GREEN,
    "pool": BLUE,
    "neg": RED,
    "point": "#8A9097",
}


def refit() -> dict:
    import pandas as pd
    import resolve_core as rc

    plots = pd.read_csv(EXAMPLE_DATA / "plots.csv")
    species = pd.read_csv(EXAMPLE_DATA / "species.csv")

    roles = rc.RoleMapping()
    roles.plot_id = "plot_id"
    roles.species_id = "species"
    roles.abundance = "cover"
    roles.genus = "genus"
    roles.family = "family"
    targets = [
        rc.TargetSpec.regression("elevation"),
        rc.TargetSpec.classification("habitat", len(HABITATS)),
    ]
    dcfg = rc.DatasetConfig()
    dcfg.species_encoding = rc.SpeciesEncodingMode.RankPool
    dataset = rc.ResolveDataset.from_pandas(plots, species, roles, targets, dcfg)

    codes = dataset.targets["habitat"].tolist()
    by_plot = dict(zip(plots.plot_id, plots.habitat))
    for pid, code in zip(dataset.plot_ids, codes):
        if HABITATS[code] != by_plot[pid]:
            raise RuntimeError("habitat codes are not in sorted label order")

    model = rc.ResolveModel(dataset.schema, rc.model_config_for_dataset(dataset))
    tcfg = rc.TrainConfig()
    tcfg.device = "cpu"
    tcfg.max_epochs = 400
    tcfg.patience = 80
    trainer = rc.Trainer(model, tcfg)
    trainer.prepare_data(dataset)
    result = trainer.fit()

    checkpoint = FIT_CACHE.with_suffix(".pt")
    trainer.save(str(checkpoint))
    out = rc.Predictor.load(str(checkpoint), device="cpu").predict_dataset(
        dataset, return_latent=True
    )
    checkpoint.unlink()

    ids = list(out.plot_ids)
    row = {pid: i for i, pid in enumerate(ids)}
    held_out = list(trainer.test_plot_ids())
    elevation = out.predictions["elevation"].numpy()
    habitat = out.predictions["habitat"].numpy()
    probabilities = out.probabilities["habitat"].numpy()
    latent = out.latent.numpy()

    richness = species.groupby("plot_id").size()

    def score(pid: str) -> tuple:
        i = row[pid]
        correct = HABITATS[habitat[i]] == by_plot[pid]
        err = abs(elevation[i] - plots.set_index("plot_id").elevation[pid])
        return (not correct, -richness[pid], err)

    shown = min(held_out, key=score)
    plot_species = species[species.plot_id == shown].sort_values("cover", ascending=False)
    observed = plots.set_index("plot_id").elevation

    return {
        "plot": shown,
        "species": [
            {"name": r.species, "family": r.family, "cover": float(r.cover)}
            for r in plot_species.itertuples()
        ],
        "latent": [round(float(x), 4) for x in latent[row[shown]]],
        "habitat_observed": by_plot[shown],
        "habitat_probabilities": {
            h: round(float(p), 4) for h, p in zip(HABITATS, probabilities[row[shown]])
        },
        "held_out": [
            {
                "plot": pid,
                "observed": float(observed[pid]),
                "predicted": round(float(elevation[row[pid]]), 1),
            }
            for pid in held_out
        ],
        "metrics": {
            "elevation_r2": round(result.final_metrics["elevation"]["r2"], 3),
            "elevation_mae": round(result.final_metrics["elevation"]["mae"], 1),
            "habitat_accuracy": round(result.final_metrics["habitat"]["accuracy"], 3),
        },
        "resolve_core": rc.__version__,
    }


def text(x, y, s, size, fill, anchor="start", weight="normal", style="normal"):
    return (
        f'<text x="{fmt(x)}" y="{fmt(y)}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" font-style="{style}">{s}</text>'
    )


def species_panel(fit: dict, th: dict) -> tuple[str, list[tuple[float, float, float]]]:
    rows = fit["species"]
    n = len(rows)
    step = ROW_STEP
    top = H / 2 - step * (n - 1) / 2
    max_cover = max(r["cover"] for r in rows)
    out = []
    anchors = []
    for k, r in enumerate(rows):
        y = top + k * step
        length = BAR_MAX * r["cover"] / max_cover
        out.append(text(MARGIN, y + 11, r["name"], LABEL, th["text"], style="italic"))
        out.append(
            f'<rect x="{fmt(BAR_X)}" y="{fmt(y - 11)}" width="{fmt(length)}" height="22" rx="4" fill="{th["plant"]}"/>'
        )
        anchors.append((BAR_X + BAR_MAX + 30, y, r["cover"] / max_cover))
    return "".join(out), anchors


def latent_column(fit: dict, th: dict) -> tuple[str, float, float, float]:
    values = fit["latent"]
    n = len(values)
    w = 34.0
    x = (BAR_X + BAR_MAX + 30 + RIGHT_X - 40) / 2 - w / 2
    height = ROW_STEP * (len(fit["species"]) - 1) + 40
    top = H / 2 - height / 2
    cell = height / n
    scale = max(abs(v) for v in values) or 1.0
    out = [
        f'<rect x="{fmt(x)}" y="{fmt(top)}" width="{fmt(w)}" height="{fmt(height)}" fill="{th["background"]}"/>'
    ]
    for k, v in enumerate(values):
        a = min(abs(v) / scale, 1.0)
        colour = th["pool"] if v >= 0 else th["neg"]
        out.append(
            f'<rect x="{fmt(x)}" y="{fmt(top + k * cell)}" width="{fmt(w)}" height="{fmt(cell + 0.3)}" '
            f'fill="{colour}" fill-opacity="{0.12 + 0.88 * a:.2f}"/>'
        )
    out.append(
        f'<rect x="{fmt(x)}" y="{fmt(top)}" width="{fmt(w)}" height="{fmt(height)}" '
        f'fill="none" stroke="{th["muted"]}" stroke-width="2" rx="3"/>'
    )
    return "".join(out), x, x + w, H / 2


def pooling_curves(anchors, x_in: float, y_mid: float, th: dict) -> str:
    out = []
    for x0, y0, weight in anchors:
        y1 = y_mid + (y0 - y_mid) * 0.12
        mid = (x0 + x_in) / 2
        out.append(
            f'<path d="M{fmt(x0)} {fmt(y0)}C{fmt(mid)} {fmt(y0)} {fmt(mid)} {fmt(y1)} {fmt(x_in)} {fmt(y1)}" '
            f'stroke-width="{fmt(2 + 12 * weight)}"/>'
        )
    return (
        f'<g fill="none" stroke="{th["plant"]}" stroke-opacity="0.45" stroke-linecap="round">'
        + "".join(out)
        + "</g>"
    )


def readout_curve(x0, y0, x1, y1, th) -> str:
    mid = (x0 + x1) / 2
    return (
        f'<path d="M{fmt(x0)} {fmt(y0)}C{fmt(mid)} {fmt(y0)} {fmt(mid)} {fmt(y1)} {fmt(x1)} {fmt(y1)}" '
        f'fill="none" stroke="{th["pool"]}" stroke-width="5" stroke-opacity="0.6" stroke-linecap="round"/>'
    )


def elevation_panel(fit: dict, th: dict) -> tuple[str, tuple[float, float]]:
    rows_top = H / 2 - ROW_STEP * (len(fit["species"]) - 1) / 2
    x0, y0, width, height = RIGHT_X, rows_top + 26, RIGHT_W, 180.0
    lo, hi = 300.0, 2700.0

    def sx(v):
        return x0 + width * (v - lo) / (hi - lo)

    def sy(v):
        return y0 + height * (1 - (v - lo) / (hi - lo))

    out = [
        f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(width)}" height="{fmt(height)}" fill="none" stroke="{th["rule"]}" stroke-width="2"/>',
        f'<path d="M{fmt(sx(lo))} {fmt(sy(lo))}L{fmt(sx(hi))} {fmt(sy(hi))}" stroke="{th["rule"]}" stroke-width="2" stroke-dasharray="8 7"/>',
    ]
    out.append(text(x0, rows_top + 4, "elevation", TITLE, th["text"]))
    shown = None
    for p in fit["held_out"]:
        cx, cy = sx(p["observed"]), sy(p["predicted"])
        if p["plot"] == fit["plot"]:
            shown = (cx, cy)
            continue
        out.append(f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="7" fill="{th["point"]}" fill-opacity="0.8"/>')
    out.append(
        f'<circle cx="{fmt(shown[0])}" cy="{fmt(shown[1])}" r="11" fill="{th["pool"]}"/>'
        f'<circle cx="{fmt(shown[0])}" cy="{fmt(shown[1])}" r="20" fill="none" stroke="{th["pool"]}" stroke-width="4"/>'
    )
    return "".join(out), (x0 - 40, y0 + height / 2)


def habitat_panel(fit: dict, th: dict) -> tuple[str, tuple[float, float]]:
    step = 44.0
    rows_bottom = H / 2 + ROW_STEP * (len(fit["species"]) - 1) / 2
    top = rows_bottom - 3 * step
    x_label, x_bar = RIGHT_X, RIGHT_X + 160
    length = RIGHT_X + RIGHT_W - x_bar
    probs = fit["habitat_probabilities"]
    best = max(probs, key=probs.get)
    out = []
    for k, h in enumerate(HABITATS):
        y = top + k * step
        p = probs[h]
        colour = th["pool"] if h == best else th["point"]
        out.append(text(x_label, y + 10, h, LABEL, th["text"]))
        out.append(f'<rect x="{fmt(x_bar)}" y="{fmt(y - 12)}" width="{fmt(length)}" height="24" rx="4" fill="{th["rule"]}" fill-opacity="0.5"/>')
        out.append(f'<rect x="{fmt(x_bar)}" y="{fmt(y - 12)}" width="{fmt(max(length * p, 3))}" height="24" rx="4" fill="{colour}"/>')
    out.append(text(x_label, top - 48, "habitat", TITLE, th["text"]))
    return "".join(out), (x_label - 40, top + step * 1.5)


def build(fit: dict) -> str:
    th = THEME
    species, anchors = species_panel(fit, th)
    latent, x_in, x_out, y_mid = latent_column(fit, th)
    elevation, elev_anchor = elevation_panel(fit, th)
    habitat, hab_anchor = habitat_panel(fit, th)
    shown = next(p for p in fit["held_out"] if p["plot"] == fit["plot"])
    m = fit["metrics"]
    alt = (
        f"Species and cover of held-out example plot {fit['plot']} pooled into one latent vector, "
        f"read out as elevation (imputed {shown['predicted']:.0f} m, observed {shown['observed']:.0f} m; "
        f"R squared {m['elevation_r2']:.2f} over {len(fit['held_out'])} held-out plots) and habitat "
        f"(observed {fit['habitat_observed']}, probability {fit['habitat_probabilities'][fit['habitat_observed']]:.2f})"
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="{alt}" font-family="{SANS}">
<title>{alt}</title>
<rect width="{W}" height="{H}" fill="{th["background"]}"/>
{pooling_curves(anchors, x_in, y_mid, th)}
{readout_curve(x_out, y_mid, elev_anchor[0], elev_anchor[1], th)}
{readout_curve(x_out, y_mid, hab_anchor[0], hab_anchor[1], th)}
{species}
{elevation}
{habitat}
{latent}
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refit", action="store_true", help="retrain and rewrite the cached fit")
    args = parser.parse_args()
    if args.refit or not FIT_CACHE.exists():
        fit = refit()
        FIT_CACHE.write_text(json.dumps(fit, indent=1) + "\n", encoding="utf-8", newline="\n")
    fit = json.loads(FIT_CACHE.read_text(encoding="utf-8"))
    R_FIGURES.mkdir(parents=True, exist_ok=True)
    path = R_FIGURES / "resolve-hero.svg"
    path.write_text(build(fit), encoding="utf-8", newline="\n")
    print(f"wrote {path}")
    m = fit["metrics"]
    print(f"plot {fit['plot']}, R2 {m['elevation_r2']}, accuracy {m['habitat_accuracy']}")


if __name__ == "__main__":
    main()
