"""High-level gap fitting for a single model.

``read_gap_tables`` reads what gap fitting needs about a dataset's plots from
pandas DataFrames or CSV paths, and ``fit_prediction_gaps`` fits a model's
targets from its predictions and those tables, as ``resolve.fit_prediction_gaps``
does in R. Both dispatch to the low-level ``read_gap_tables_csv`` /
``read_gap_tables_columns`` / ``fit_prediction_gaps_tables`` bindings.
"""

from __future__ import annotations

import os

from . import _resolve_core as _core
from ._from_pandas import _df_to_columns
from ._suite import _gap_fitting, _recorded

_COLUMN_KEYS = ("plot_id", "species_id", "latitude", "longitude", "survey_year", "recorded")
_SCALES = {
    "linear": "Linear",
    "log": "Log",
    "circular": "Circular",
    "geographic": "Geographic",
}


def _gap_columns(columns) -> "_core.GapColumns":
    if isinstance(columns, _core.GapColumns):
        return columns
    if not isinstance(columns, dict):
        raise TypeError("columns must be a GapColumns or a dict")
    unknown = sorted(set(columns) - set(_COLUMN_KEYS))
    if unknown:
        raise ValueError(f"columns: unknown key(s) {unknown}; accepted: {list(_COLUMN_KEYS)}")
    out = _core.GapColumns()
    for key, value in columns.items():
        setattr(out, key, _recorded(value) if key == "recorded" else value)
    return out


def read_gap_tables(species, header=None, columns=None, plot_ids=None):
    """What gap fitting reads about the plots ``plot_ids``, in that order.

    Parameters
    ----------
    species : pandas.DataFrame or str or os.PathLike
        The species records, one row per record, or a CSV path.
    header : pandas.DataFrame or str or os.PathLike or None
        The plots, one row per plot, of the same kind as ``species``; None
        reads plot-level columns from each plot's first species row.
    columns : GapColumns or dict
        ``plot_id``, ``species_id``, optionally ``latitude``, ``longitude``,
        ``survey_year``, and ``recorded`` (``{target: column}``, or
        ``{target: [latitude, longitude]}`` for a location).
    plot_ids : sequence of str
        The plots, in the order of the prediction to fit.

    Returns
    -------
    GapTables
    """
    cols = _gap_columns(columns)
    ids = [str(p) for p in plot_ids]
    if isinstance(species, (str, os.PathLike)):
        if header is not None and not isinstance(header, (str, os.PathLike)):
            raise TypeError("with a species CSV path, header must be a CSV path or None")
        return _core.read_gap_tables_csv(
            os.fspath(species), "" if header is None else os.fspath(header), cols, ids
        )
    s_names, s_cols = _df_to_columns(species, "species")
    if header is None:
        return _core.read_gap_tables_columns(s_names, s_cols, None, None, cols, ids)
    h_names, h_cols = _df_to_columns(header, "header")
    return _core.read_gap_tables_columns(s_names, s_cols, h_names, h_cols, cols, ids)


def _request(name, spec) -> "_core.GapFitRequest":
    spec = dict(spec or {})
    unknown = sorted(set(spec) - {"scale", "period", "outputs"})
    if unknown:
        raise ValueError(f"targets[{name!r}]: unknown key(s) {unknown}")
    scale = spec.get("scale", "linear")
    if isinstance(scale, str):
        if scale not in _SCALES:
            raise ValueError(f"targets[{name!r}]: unknown scale {scale!r}; accepted: {list(_SCALES)}")
        scale = getattr(_core.GapScale, _SCALES[scale])
    target = _core.GapTarget(str(name), scale, float(spec.get("period", 360.0)))
    outputs = spec.get("outputs", [name])
    outputs = [outputs] if isinstance(outputs, str) else [str(o) for o in outputs]
    return _core.GapFitRequest(target, outputs)


def fit_prediction_gaps(predictions, species, header=None, columns=None, targets=None,
                        gap_fitting=None):
    """Fit a single model's gaps from its predictions and the dataset's tables.

    Parameters
    ----------
    predictions : ResolvePredictions
        ``Predictor.predict_dataset`` output.
    species, header, columns
        As for ``read_gap_tables``; the tables are read in the order of
        ``predictions.plot_ids``.
    targets : dict
        ``{target: {"scale": ..., "period": ..., "outputs": [...]}}``: the
        scale (``"linear"``, ``"log"``, ``"circular"``, ``"geographic"``) and
        the prediction head(s) read (default the target's name; a bearing may
        read its sine and cosine heads, a location its latitude and longitude
        heads).
    gap_fitting : GapFittingConfig or dict or None
        The settings (``candidates``, ``k_grid``, ``spatial``,
        ``matching_seed``); None for the defaults.

    Returns
    -------
    dict of str to GapFit
    """
    if not targets:
        raise ValueError("targets must name at least one target to fit")
    tables = read_gap_tables(species, header, columns, list(predictions.plot_ids))
    requests = [_request(name, spec) for name, spec in targets.items()]
    config = _gap_fitting(True if gap_fitting is None else gap_fitting)
    fits = _core.fit_prediction_gaps_tables(predictions, tables, requests, config)
    return {r.target.name: f for r, f in zip(requests, fits)}
