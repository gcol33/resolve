"""Name resolution over Python inputs.

``species_names`` turns the ``names`` argument of the prediction front doors
into the engine's ``SpeciesNames``; ``resolve_names`` resolves a list of names
against a model's vocabulary; ``SpeciesNameReport.to_pandas`` is a report as a
DataFrame.
"""

from __future__ import annotations

import os

from . import _resolve_core as _core


def species_names(names, max_edit_distance=1):
    """The engine's ``SpeciesNames`` for a ``names`` argument.

    ``None`` or ``False`` reads every name as written. A path (str or
    os.PathLike) is the release's ``names.json``, read once per process; a
    ``NameTable`` is used as given; ``True`` resolves exact names and
    spellings against the vocabulary alone. A ``SpeciesNames`` is returned
    unchanged and ``max_edit_distance`` is then ignored.
    """
    if names is None or names is False:
        return None
    if isinstance(names, _core.SpeciesNames):
        return names
    if names is True:
        return _core.SpeciesNames(None, max_edit_distance)
    if isinstance(names, (str, os.PathLike)):
        return _core.SpeciesNames(
            _core.NameTable.open(os.fspath(names)), max_edit_distance
        )
    if isinstance(names, _core.NameTable):
        return _core.SpeciesNames(names, max_edit_distance)
    raise TypeError(
        "names must be a path to names.json, a NameTable, a SpeciesNames, True or None"
    )


def _vocabulary(model, target):
    if isinstance(model, _core.SuitePredictor):
        if target is None:
            raise ValueError("a suite holds one vocabulary per target: pass target=")
        return model.species_vocabulary(target)
    if isinstance(model, _core.Predictor):
        return list(model.external_vocabs.species_vocab)
    return list(model)


def resolve_names(names, model, target=None, table=True, max_edit_distance=1):
    """Resolve species names against a model's vocabulary.

    Parameters
    ----------
    names : sequence of str
        The species names as written.
    model : SuitePredictor or Predictor or sequence of str
        Where the vocabulary comes from: a suite target's (``target``), a
        single model's, or a vocabulary as a checkpoint orders it
        (``[0] = "<UNK>"``).
    target : str, optional
        The suite target whose vocabulary is used.
    table : str or os.PathLike or NameTable or None
        The release's ``names.json`` (a path) or a ``NameTable``; ``None`` or
        ``True`` resolves against the vocabulary alone.
    max_edit_distance : int
        The edits a misspelled name may be away from a known name; 0 turns
        the spelling step off.

    Returns
    -------
    pandas.DataFrame
        One row per name, in order: ``name``, ``token``, ``route`` (exact,
        alias, spelling or unresolved), ``source``, ``via``, ``distance``.
    """
    import pandas as pd

    setting = species_names(True if table is None else table, max_edit_distance)
    resolver = _core.NameResolver(
        _vocabulary(model, target), setting.table, setting.max_edit_distance
    )
    names = list(names)
    resolved = resolver.resolve_batch(names)
    return pd.DataFrame(
        {
            "name": names,
            "token": [r.token for r in resolved],
            "route": [r.route.name.lower() for r in resolved],
            "source": [r.source for r in resolved],
            "via": [r.via for r in resolved],
            "distance": [r.distance for r in resolved],
        }
    )


def report_to_pandas(self):
    """The report as a DataFrame: name, token, route, source, via, distance,
    n_records."""
    import pandas as pd

    return pd.DataFrame(
        {
            "name": self.name,
            "token": self.token,
            "route": self.route,
            "source": self.source,
            "via": self.via,
            "distance": self.distance,
            "n_records": self.n_records,
        }
    )


def install():
    """Attach to_pandas to SpeciesNameReport."""
    _core.SpeciesNameReport.to_pandas = report_to_pandas
