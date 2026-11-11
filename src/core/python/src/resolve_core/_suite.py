"""High-level model-suite prediction.

``SuitePredictor.predict`` scores plots given as pandas DataFrames or as CSV
paths, dispatching to the low-level ``predict_columns`` / ``predict_csv``
bindings; ``SuitePredictions.to_pandas`` is the engine's prediction table
(``suite_prediction_columns``) as a DataFrame, the columns
``resolve predict --suite`` writes.
"""

from __future__ import annotations

import os

from . import _resolve_core as _core
from ._from_pandas import _df_to_columns
from ._names import species_names


def _columns(columns):
    """A SuiteColumns from a SuiteColumns, a dict of its fields, or None."""
    if columns is None:
        return _core.SuiteColumns()
    if isinstance(columns, _core.SuiteColumns):
        return columns
    if not isinstance(columns, dict):
        raise TypeError("columns must be a SuiteColumns or a dict")
    out = _core.SuiteColumns()
    fields = [name for name in dir(out) if not name.startswith("_")]
    unknown = sorted(set(columns) - set(fields))
    if unknown:
        raise ValueError(f"columns: unknown key(s) {unknown}; accepted: {fields}")
    for key, value in columns.items():
        if key in ("covariates", "categoricals") and value is not None:
            value = [str(v) for v in value]
        if key == "recorded":
            value = _recorded(value)
        if key == "terrain":
            value = _terrain_columns(value)
        setattr(out, key, value)
    return out


def _terrain_columns(value):
    """SuiteTerrainColumns from itself or a dict of elevation, elevation_1km,
    slope, slope_1km and uncertainty column names."""
    if isinstance(value, _core.SuiteTerrainColumns):
        return value
    if not isinstance(value, dict):
        raise TypeError("columns['terrain'] must be a SuiteTerrainColumns or a dict")
    out = _core.SuiteTerrainColumns()
    fields = ["elevation", "elevation_1km", "slope", "slope_1km", "uncertainty"]
    unknown = sorted(set(value) - set(fields))
    if unknown:
        raise ValueError(f"columns['terrain']: unknown key(s) {unknown}; accepted: {fields}")
    for key, column in value.items():
        setattr(out, key, None if column is None else str(column))
    return out


_TERRAIN_SOURCES = {"field": "Field", "grids": "Grids", "columns": "Columns"}


def _terrain(terrain):
    """SuiteTerrainOptions from None, True (the suite's field), a source name
    ('field', 'grids', 'columns'), a dict of source / grids / device, or the
    options themselves; None and False correct nothing."""
    if terrain is None or terrain is False:
        return None
    if isinstance(terrain, _core.SuiteTerrainOptions):
        return terrain
    if terrain is True:
        return _core.SuiteTerrainOptions()
    if isinstance(terrain, str):
        terrain = {"source": terrain}
    if not isinstance(terrain, dict):
        raise TypeError("terrain must be SuiteTerrainOptions, a source name, a dict, True or None")
    unknown = sorted(set(terrain) - {"source", "grids", "device"})
    if unknown:
        raise ValueError(f"terrain: unknown key(s) {unknown}; accepted: ['device', 'grids', 'source']")
    source = terrain.get("source", "field")
    if isinstance(source, str):
        if source not in _TERRAIN_SOURCES:
            raise ValueError(f"terrain: unknown source '{source}'; accepted: {sorted(_TERRAIN_SOURCES)}")
        source = getattr(_core.TerrainSourceKind, _TERRAIN_SOURCES[source])
    grids = terrain.get("grids", "")
    return _core.SuiteTerrainOptions(source, "" if grids is None else os.fspath(grids), terrain.get("device", "cpu"))


def _recorded(value):
    """The recorded columns as (target, [column, ...]) pairs, from a
    dict or pairs whose column is one name or a list of names."""
    items = value.items() if isinstance(value, dict) else value
    out = []
    for target, cols in items:
        out.append((str(target), [str(cols)] if isinstance(cols, str) else [str(c) for c in cols]))
    return out


def _gap_fitting(gap_fitting):
    """A GapFittingConfig from None, True, a GapFittingConfig or a dict of its
    fields; None and False fit nothing."""
    if gap_fitting is None or gap_fitting is False:
        return None
    if gap_fitting is True:
        return _core.GapFittingConfig()
    if isinstance(gap_fitting, _core.GapFittingConfig):
        return gap_fitting
    if not isinstance(gap_fitting, dict):
        raise TypeError("gap_fitting must be a GapFittingConfig, a dict, True or None")
    out = _core.GapFittingConfig()
    fields = [name for name in dir(out) if not name.startswith("_") and name != "validate"]
    unknown = sorted(set(gap_fitting) - set(fields))
    if unknown:
        raise ValueError(f"gap_fitting: unknown key(s) {unknown}; accepted: {fields}")
    for key, value in gap_fitting.items():
        if key == "candidates":
            value = [getattr(_core.GapMethod, _METHODS[v]) if isinstance(v, str) else v for v in value]
        setattr(out, key, value)
    return out


# The engine's method names (gap_method_label) and their enum members.
_METHODS = {
    "local_constant": "LocalConstant",
    "resolve": "Resolve",
    "offset_slope": "OffsetSlope",
    "knn_species": "KnnSpecies",
    "stack_species": "StackSpecies",
    "knn_spatial": "KnnSpatial",
    "stack_species_spatial": "StackSpeciesSpatial",
}


def predict(
    self,
    species,
    header=None,
    columns=None,
    batch_size=4096,
    keep_members=False,
    gap_fitting=None,
    names=None,
    max_edit_distance=1,
    terrain=None,
):
    """Score plots with every loaded target of the suite.

    Parameters
    ----------
    species : pandas.DataFrame or str or os.PathLike
        The species records, one row per record, or a CSV path.
    header : pandas.DataFrame or str or os.PathLike or None
        The plots, one row per plot. Required when the suite reads plot-level
        columns (coordinates or covariates); when given it decides which plots
        are scored and their order. Must be the same kind as ``species``.
    columns : SuiteColumns or dict, optional
        Renames the manifest's input columns (keys ``plot_id``, ``species_id``,
        ``abundance``, ``genus``, ``family``, ``latitude``, ``longitude``,
        ``covariates``, ``categoricals``), ``recorded`` (``{target: column}``,
        ``{"location": [lat, lon]}``: the column holding each regression
        target's recorded value, which adds ``<target>_recorded``, or the gap
        fit with ``gap_fitting``), for gap fitting ``survey_year``, and for
        terrain correction ``terrain``
        (``{"elevation": col, "elevation_1km": col, "slope": col,
        "slope_1km": col, "uncertainty": col}``).
    batch_size : int
        Per-member forward chunk (-1 for one pass).
    keep_members : bool
        Keep every member's own prediction in ``members``.
    gap_fitting : GapFittingConfig or dict or bool, optional
        Fit each target ``columns["recorded"]`` names from the dataset's own
        complete plots (``True`` for the defaults; a dict sets
        ``candidates``, ``k_grid``, ``spatial``, ``matching_seed``,
        ``n_folds``, ``max_fold_share``, ``matching_sample``). Adds
        ``<target>_fitted`` and ``<target>_fit_method`` to ``to_pandas`` and
        a ``gap_fit`` (with its ``report``) to each fitted target.
    names : str or os.PathLike or NameTable or SpeciesNames or bool, optional
        Resolve the species names against each target's vocabulary before
        encoding: the release's ``names.json`` (a path) or a ``NameTable``
        maps entered names and synonyms to the tokens, ``True`` uses the
        vocabulary alone. Each target's ``species_names`` then reports the
        names not found as written. ``None`` reads every name as written.
    max_edit_distance : int
        With ``names``: the edits a misspelled name may be away from a known
        name (default 1; 0 turns the spelling step off; 2 can also rename
        species the models never saw).
    terrain : str or dict or SuiteTerrainOptions or bool, optional
        Correct every target the suite gives a terrain correction (altitude,
        slope) by an elevation model read at the plots: ``"field"`` (or
        ``True``) reads the suite's terrain field at the coordinates,
        ``{"source": "grids", "grids": dir}`` elevation grids,
        ``"columns"`` your own values named in ``columns["terrain"]``. Adds
        ``<target>_terrain``, the weight of each source and the descriptors
        read to ``to_pandas``, and a ``terrain`` result to each corrected
        target.

    Returns
    -------
    SuitePredictions
    """
    cols = _columns(columns)
    resolve = species_names(names, max_edit_distance)
    if isinstance(species, (str, os.PathLike)):
        if header is not None and not isinstance(header, (str, os.PathLike)):
            raise TypeError(
                "with a species CSV path, header must be a CSV path or None"
            )
        return self.predict_csv(
            os.fspath(species),
            "" if header is None else os.fspath(header),
            cols,
            batch_size,
            keep_members,
            _gap_fitting(gap_fitting),
            resolve,
            _terrain(terrain),
        )
    s_names, s_cols = _df_to_columns(species, "species")
    if header is None:
        return self.predict_columns(
            s_names,
            s_cols,
            None,
            None,
            cols,
            batch_size,
            keep_members,
            _gap_fitting(gap_fitting),
            resolve,
            _terrain(terrain),
        )
    h_names, h_cols = _df_to_columns(header, "header")
    return self.predict_columns(
        s_names,
        s_cols,
        h_names,
        h_cols,
        cols,
        batch_size,
        keep_members,
        _gap_fitting(gap_fitting),
        resolve,
        _terrain(terrain),
    )


def to_pandas(self, probabilities=False):
    """One row per plot, with the columns ``resolve predict --suite`` writes.

    ``plot_id``, then per target the prediction (``<target>``; for a vote also
    ``<target>_code`` and ``<target>_agreement``, and with ``probabilities``
    ``<target>_prob_<class>``; for a spherical mean ``<target>_latitude`` and
    ``<target>_longitude``), its dispersion (``<target>_sd`` for a mean,
    ``<target>_circular_sd`` for a bearing, ``<target>_spread_km`` for a
    spherical mean), ``<target>_imputed`` (bool, False where none of the
    plot's species is recognised, so every value of the plot is missing), the
    four recognition columns, and ``<target>_seed<N>``
    per member when the members were kept (``<target>_seed<N>_latitude`` and
    ``_longitude`` for a spherical mean); a gap-fitted target adds
    ``<target>_fitted`` (``_fitted_latitude`` / ``_fitted_longitude``) and
    ``<target>_fit_method``, and a target with recorded values but no gap fit
    ``<target>_recorded`` (bool) there instead; a terrain-corrected target
    ``<target>_terrain``, ``<target>_terrain_w_resolve``, ``_w_dem``,
    ``_w_dem_1km``, ``<target>_dem`` and ``<target>_dem_1km``. The table is the
    engine's
    (``suite_prediction_columns``).
    """
    import pandas as pd

    # An empty text cell is a class label of a plot that was not imputed.
    return pd.DataFrame(
        {
            name: [v if v != "" else None for v in values] if isinstance(values, list) else values.numpy()
            for name, values in _core.suite_prediction_columns(self, probabilities)
        }
    )


def install():
    """Attach predict / to_pandas to the suite bindings."""
    _core.SuitePredictor.predict = predict
    _core.SuitePredictions.to_pandas = to_pandas
