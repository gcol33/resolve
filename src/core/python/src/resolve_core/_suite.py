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
        setattr(out, key, value)
    return out


def predict(
    self, species, header=None, columns=None, batch_size=4096, keep_members=False
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
        ``covariates``, ``categoricals``).
    batch_size : int
        Per-member forward chunk (-1 for one pass).
    keep_members : bool
        Keep every member's own prediction in ``members``.

    Returns
    -------
    SuitePredictions
    """
    cols = _columns(columns)
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
        )
    s_names, s_cols = _df_to_columns(species, "species")
    if header is None:
        return self.predict_columns(
            s_names, s_cols, None, None, cols, batch_size, keep_members
        )
    h_names, h_cols = _df_to_columns(header, "header")
    return self.predict_columns(
        s_names, s_cols, h_names, h_cols, cols, batch_size, keep_members
    )


def to_pandas(self, probabilities=False):
    """One row per plot, with the columns ``resolve predict --suite`` writes.

    ``plot_id``, then per target the prediction (``<target>``; for a vote also
    ``<target>_code`` and ``<target>_agreement``, and with ``probabilities``
    ``<target>_prob_<class>``; for a spherical mean ``<target>_latitude`` and
    ``<target>_longitude``), its dispersion (``<target>_sd`` for a mean,
    ``<target>_circular_sd`` for a bearing, ``<target>_spread_km`` for a
    spherical mean), the four recognition columns, and ``<target>_seed<N>``
    per member when the members were kept (``<target>_seed<N>_latitude`` and
    ``_longitude`` for a spherical mean). The table is the engine's
    (``suite_prediction_columns``).
    """
    import pandas as pd

    return pd.DataFrame(
        {
            name: values if isinstance(values, list) else values.numpy()
            for name, values in _core.suite_prediction_columns(self, probabilities)
        }
    )


def install():
    """Attach predict / to_pandas to the suite bindings."""
    _core.SuitePredictor.predict = predict
    _core.SuitePredictions.to_pandas = to_pandas
