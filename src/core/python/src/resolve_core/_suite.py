"""High-level model-suite prediction.

``SuitePredictor.predict`` scores plots given as pandas DataFrames or as CSV
paths, dispatching to the low-level ``predict_columns`` / ``predict_csv``
bindings; ``SuitePredictions.to_pandas`` flattens a result to one row per plot,
with the columns ``resolve predict --suite`` writes.
"""
from __future__ import annotations

import os

from . import _resolve_core as _core
from ._from_pandas import _df_to_columns
from ._io_retry import retry_io

_COLUMN_FIELDS = ("plot_id", "species_id", "abundance", "genus", "family",
                  "latitude", "longitude", "covariates", "categoricals")


def _columns(columns):
    """A SuiteColumns from a SuiteColumns, a dict of its fields, or None."""
    if columns is None:
        return _core.SuiteColumns()
    if isinstance(columns, _core.SuiteColumns):
        return columns
    if not isinstance(columns, dict):
        raise TypeError("columns must be a SuiteColumns or a dict")
    unknown = sorted(set(columns) - set(_COLUMN_FIELDS))
    if unknown:
        raise ValueError(f"columns: unknown key(s) {unknown}; accepted: {list(_COLUMN_FIELDS)}")
    out = _core.SuiteColumns()
    for key, value in columns.items():
        if key in ("covariates", "categoricals") and value is not None:
            value = [str(v) for v in value]
        setattr(out, key, value)
    return out


def predict(self, species, header=None, columns=None, batch_size=4096, keep_members=False):
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
            raise TypeError("with a species CSV path, header must be a CSV path or None")
        return self.predict_csv(os.fspath(species),
                                "" if header is None else os.fspath(header),
                                cols, batch_size, keep_members)
    s_names, s_cols = _df_to_columns(species, "species")
    if header is None:
        return self.predict_columns(s_names, s_cols, None, None, cols, batch_size, keep_members)
    h_names, h_cols = _df_to_columns(header, "header")
    return self.predict_columns(s_names, s_cols, h_names, h_cols, cols, batch_size,
                                keep_members)


def to_pandas(self, probabilities=False):
    """One row per plot, with the columns ``resolve predict --suite`` writes.

    ``plot_id``, then per target the prediction (``<target>``; for a vote also
    ``<target>_code`` and ``<target>_agreement``, and with ``probabilities``
    ``<target>_prob_<class>``), its dispersion (``<target>_sd`` for a mean,
    ``<target>_circular_sd`` for a bearing), the four recognition columns, and
    ``<target>_seed<N>`` per member when the members were kept.
    """
    import pandas as pd

    data = {"plot_id": list(self.plot_ids)}
    for t in self.targets:
        name = t.name
        names = list(t.class_names)
        if t.combine == _core.SuiteCombine.Vote:
            codes = t.value.tolist()
            data[name] = [names[c] if 0 <= c < len(names) else str(c) for c in codes]
            data[f"{name}_code"] = codes
            data[f"{name}_agreement"] = t.agreement.numpy()
            if probabilities:
                probs = t.probabilities.numpy()
                for k in range(probs.shape[1]):
                    label = names[k] if k < len(names) else str(k)
                    data[f"{name}_prob_{label}"] = probs[:, k]
        else:
            data[name] = t.value.numpy()
            suffix = "_circular_sd" if t.combine == _core.SuiteCombine.CircularMean else "_sd"
            data[f"{name}{suffix}"] = t.dispersion.numpy()
        r = t.recognition
        data[f"{name}_n_species"] = r.n_species.numpy()
        data[f"{name}_n_recognised"] = r.n_recognised.numpy()
        data[f"{name}_recognised_share"] = r.count_share.numpy()
        data[f"{name}_recognised_abundance_share"] = r.abundance_share.numpy()
        members = t.members
        if members is not None:
            for k, seed in enumerate(t.member_seeds):
                row = members[k].tolist()
                if t.combine == _core.SuiteCombine.Vote:
                    row = [names[c] if 0 <= c < len(names) else str(c) for c in row]
                data[f"{name}_seed{seed}"] = row
    return pd.DataFrame(data)


def install():
    """Attach predict / to_pandas, and retry transient storage faults on load."""
    _load = _core.SuitePredictor.load

    def load(dir, device="cpu", vram_fraction=1.0, verify=True, targets=None):
        return retry_io(lambda: _load(dir, device, vram_fraction, verify, targets),
                        what=f"SuitePredictor.load({dir!r})")

    load.__doc__ = _load.__doc__
    _core.SuitePredictor.load = staticmethod(load)
    _core.SuitePredictor.predict = predict
    _core.SuitePredictions.to_pandas = to_pandas
