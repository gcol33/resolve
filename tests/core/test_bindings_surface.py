"""The public surface of the compiled extension is reachable from ``resolve_core``.

``resolve_core/__init__.py`` re-exports what ``_resolve_core`` registers, so user
code, the docs and this suite all import from one path. A binding added to
``src/core/python/src/bindings_*.cpp`` without a matching entry in ``__init__.py``
is reachable only through the private ``resolve_core._resolve_core`` module,
which is how ``SpatialBlockConfig``, ``CategoricalVocab`` and
``ClassificationPredictions`` ended up being imported privately by tests, docs
and benchmarks. These checks are mechanical, so the next one fails here instead.
"""

from __future__ import annotations

import enum
import importlib

import pytest

rc = pytest.importorskip("resolve_core")


def _public_names(module) -> set[str]:
    """Names a caller is meant to use: no leading underscore, no re-exported module."""
    return {name for name in dir(module) if not name.startswith("_")}


def _is_enum_member(obj) -> bool:
    """``PoolWeighting.Abundance`` is also ``_resolve_core.Abundance``.

    nanobind puts each enumerator at module level beside its enum type. The
    enumerator is reached through that type, which is itself checked below, so
    it carries no separate export obligation.
    """
    return isinstance(obj, enum.Enum)


def _is_alias(module, name: str, obj) -> bool:
    """``m.attr("SpaccModel") = m.attr("ResolveModel")`` -- a second name for a
    type already exported under its own. The object is reachable, so the alias
    is not a missing export.
    """
    canonical = getattr(obj, "__qualname__", name)
    return canonical != name and getattr(module, canonical, None) is obj


def _canonical_names(module) -> set[str]:
    """Every binding that owes ``__init__.py`` an entry of its own."""
    return {
        name
        for name in _public_names(module)
        if not _is_enum_member(getattr(module, name))
        and not _is_alias(module, name, getattr(module, name))
    }


@pytest.fixture(scope="module")
def native():
    return importlib.import_module("resolve_core._resolve_core")


def test_every_native_name_is_reexported(native):
    missing = sorted(name for name in _canonical_names(native) if not hasattr(rc, name))
    assert missing == [], (
        "registered in bindings_*.cpp but not re-exported from "
        "resolve_core/__init__.py: " + ", ".join(missing)
    )


def test_every_native_name_is_in_all(native):
    missing = sorted(name for name in _canonical_names(native) if name not in rc.__all__)
    assert missing == [], (
        "re-exported but absent from resolve_core.__all__: " + ", ".join(missing)
    )


def test_every_native_object_is_reachable(native):
    """Nothing registered is reachable only through the private module.

    Wider than the two checks above: an enumerator or an alias satisfies this by
    way of the type that owns it, so the set of objects is covered even though
    the set of names is not.
    """
    exported = {id(getattr(rc, name)) for name in rc.__all__}
    for name in sorted(_public_names(native)):
        obj = getattr(native, name)
        if _is_enum_member(obj):
            assert id(type(obj)) in exported, f"{name}: enum type {type(obj).__name__}"
        else:
            assert id(obj) in exported, name


def test_every_all_entry_resolves():
    missing = sorted(name for name in rc.__all__ if not hasattr(rc, name))
    assert missing == [], "listed in __all__ but not defined: " + ", ".join(missing)


def test_all_has_no_duplicates():
    seen = sorted({name for name in rc.__all__ if rc.__all__.count(name) > 1})
    assert seen == [], "duplicated in __all__: " + ", ".join(seen)


def test_reexport_is_the_same_object(native):
    """A re-export must alias the native object, not shadow it with a stub."""
    for name in sorted(_canonical_names(native)):
        assert getattr(rc, name) is getattr(native, name), name


@pytest.mark.parametrize(
    "name",
    [
        "SpatialBlockConfig",
        "CategoricalVocab",
        "ClassificationPredictions",
        "SpeciesVocab",
        "RankPoolEncoder",
        "EmbeddingEncoder",
        "UnknownSpeciesStats",
        "compute_unknown_species_stats",
        "ClassificationMetrics",
        "ConfidenceMetrics",
        "TabMConfig",
        "ModelForwardResult",
        "fuzzy",
    ],
)
def test_previously_private_names_are_public(name):
    assert hasattr(rc, name)


def test_fuzzy_submodule_carries_its_types():
    assert hasattr(rc.fuzzy, "FuzzyIndex")
    assert hasattr(rc.fuzzy, "Match")


def test_spatial_block_config_is_usable_from_the_public_path():
    """The import `tests/core/test_spatial.py` and the docs used to reach for."""
    cfg = rc.SpatialBlockConfig()
    cfg.lat_size = 1.5
    cfg.lon_size = 2.5
    cfg.balance = True
    assert cfg.lat_size == pytest.approx(1.5)
    assert cfg.lon_size == pytest.approx(2.5)
    assert cfg.balance is True
