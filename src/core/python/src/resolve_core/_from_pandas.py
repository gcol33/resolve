"""High-level in-memory dataset loaders (issue #22).

``ResolveDataset.from_pandas`` / ``.from_dataframe`` build a dataset directly
from pandas DataFrames already in RAM, eliminating the write-to-temp-CSV /
re-read round-trip that ``from_csv`` forces whenever the header must be filtered
or subset before a fit.

Cells are stringified exactly as ``DataFrame.to_csv`` would write them (missing
values become the empty string), and the engine reuses the identical CSV loader
body via its RowSource seam, so ``from_pandas(df)`` is equivalent to
``from_csv`` on ``df.to_csv()`` -- only the disk I/O is elided.
"""
from __future__ import annotations

from . import _resolve_core as _core

_ResolveDataset = _core.ResolveDataset


def _is_dataframe(obj) -> bool:
    # Duck-typed pandas detection: avoids importing pandas unless the caller
    # actually passes a frame (pandas is not a hard dependency of resolve_core).
    cls = type(obj)
    return (
        any(getattr(c, "__name__", "") == "DataFrame" for c in cls.__mro__)
        and hasattr(obj, "columns")
        and hasattr(obj, "__getitem__")
    )


def _df_to_columns(df, what: str):
    """Convert a DataFrame to (names, columns) of strings, NA -> "".

    Stringification matches ``DataFrame.to_csv`` (default ``na_rep=""``), so a
    dataset built from the frame is identical to one built from the CSV the
    frame would serialize to.
    """
    if not _is_dataframe(df):
        raise TypeError(
            f"{what} must be a pandas DataFrame, got {type(df).__name__}"
        )
    names = [str(c) for c in df.columns]
    columns = []
    for c in df.columns:
        col = df[c]
        s = col.astype(str)
        na = col.isna()
        # `.any()` on the NA mask is cheap and lets us skip the mask when the
        # column is dense (the common case for plot_id / species_id columns).
        if bool(na.any()):
            s = s.mask(na, "")
        columns.append(s.tolist())
    return names, columns


def from_pandas(header, species=None, roles=None, targets=None,
                config=None, schema_source=None):
    """Load a :class:`ResolveDataset` from in-memory pandas DataFrame(s).

    Parameters
    ----------
    header : pandas.DataFrame
        Header frame (one row per plot: targets, covariates, coordinates). In
        single-table mode (``species is None``) this is instead the long-format
        species frame, matching ``from_species_csv``.
    species : pandas.DataFrame or str or None, optional
        Species frame (long format: multiple rows per plot). If a ``str``, it is
        treated as a CSV path and the large species table is read once from disk
        while the header stays in memory. If ``None``, single-table mode.
    roles : RoleMapping
    targets : sequence of TargetSpec
    config : DatasetConfig, optional
    schema_source : ResolveDataset, optional
        Reuse this dataset's vocabularies / class mappings (the in-memory analog
        of ``from_csv_with_schema``). Only valid when ``species`` is a
        DataFrame.

    Returns
    -------
    ResolveDataset
    """
    if roles is None or targets is None:
        raise TypeError("from_pandas requires `roles` and `targets`")
    targets = list(targets)
    kw = {} if config is None else {"config": config}

    # Single-table mode: `header` is the long-format species frame.
    if species is None:
        if schema_source is not None:
            raise ValueError(
                "schema_source is not supported in single-table mode "
                "(pass a separate species frame to reuse a schema)"
            )
        names, cols = _df_to_columns(header, "header/species frame")
        return _ResolveDataset.from_species_columns(
            names, cols, roles, targets, **kw)

    h_names, h_cols = _df_to_columns(header, "header")

    # Header in memory + species streamed from a CSV path.
    if isinstance(species, str):
        if schema_source is not None:
            raise ValueError(
                "schema_source is not supported with a species CSV path; "
                "pass both header and species as DataFrames"
            )
        return _ResolveDataset.from_columns_header(
            h_names, h_cols, species, roles, targets, **kw)

    # Both frames in memory.
    s_names, s_cols = _df_to_columns(species, "species")
    if schema_source is not None:
        return _ResolveDataset.from_columns_with_schema(
            h_names, h_cols, s_names, s_cols, roles, targets, schema_source, **kw)
    return _ResolveDataset.from_columns(
        h_names, h_cols, s_names, s_cols, roles, targets, **kw)


def install():
    """Attach from_pandas / from_dataframe as ResolveDataset static methods."""
    _ResolveDataset.from_pandas = staticmethod(from_pandas)
    _ResolveDataset.from_dataframe = staticmethod(from_pandas)
