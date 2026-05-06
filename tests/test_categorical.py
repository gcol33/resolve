"""Tests for auto-categorical encoding in from_fast_csv and the helper.

Covers:
- _encode_categorical preserves integer-string codes verbatim.
- _encode_categorical factorizes letter codes alphabetically.
- _encode_categorical with explicit mapping (e.g. {"Y": 1, "N": 0}).
- NA-like strings always become None.
- from_fast_csv loads a CSV with letter-coded EUNIS target and Y/N covariate
  without manual preprocessing, and auto-fills num_classes.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import polars as pl
import pytest

from resolve.data.dataset import ResolveDataset, _encode_categorical


class TestEncodeCategorical:
    def test_integer_strings_preserved(self):
        codes, mapping = _encode_categorical(["0", "1", "2", "1", "0"])
        assert codes == [0, 1, 2, 1, 0]
        assert mapping == {"0": 0, "1": 1, "2": 2}

    def test_wide_integer_range_preserved(self):
        # Lex-sort would break "10" < "2"; integer parse must preserve.
        codes, mapping = _encode_categorical(["2", "10", "11", "2"])
        assert codes == [2, 10, 11, 2]
        assert mapping == {"2": 2, "10": 10, "11": 11}

    def test_letter_codes_factorized_sorted(self):
        codes, mapping = _encode_categorical(
            ["M", "N", "P", "Q", "R", "S", "T", "U", "V", "M"]
        )
        assert mapping == {
            "M": 0, "N": 1, "P": 2, "Q": 3, "R": 4,
            "S": 5, "T": 6, "U": 7, "V": 8,
        }
        assert codes == [0, 1, 2, 3, 4, 5, 6, 7, 8, 0]

    def test_explicit_mapping_applied(self):
        codes, mapping = _encode_categorical(
            ["Y", "N", "Y", "Maybe", ""], mapping={"Y": 1, "N": 0}
        )
        # Unmapped non-NA → None; NA-string → None
        assert codes == [1, 0, 1, None, None]
        assert mapping == {"Y": 1, "N": 0}

    def test_na_strings_become_none(self):
        codes, _ = _encode_categorical(["A", "NA", "", ".", "B", "NaN", "null"])
        # A=0, B=1 (sorted unique non-NA), everything else None
        assert codes == [0, None, None, None, 1, None, None]

    def test_empty_input(self):
        codes, mapping = _encode_categorical([])
        assert codes == []
        assert mapping == {}

    def test_all_na(self):
        codes, mapping = _encode_categorical(["NA", "", "NaN"])
        assert codes == [None, None, None]
        assert mapping == {}


def _write_csv(rows: list[dict], path: str) -> None:
    df = pl.DataFrame(rows)
    df.write_csv(path)


@pytest.fixture
def synthetic_csv_paths(tmp_path):
    """Build tiny header + species CSVs with a letter-coded target and Y/N covariate."""
    header_path = str(tmp_path / "header.csv")
    species_path = str(tmp_path / "species.csv")

    rows = []
    eunis_letters = ["M", "N", "P", "Q", "R", "S", "T", "U", "V"]
    rng = np.random.default_rng(0)
    for i in range(50):
        rows.append({
            "PlotObservationID": f"P{i:04d}",
            "Latitude": float(rng.uniform(45, 55)),
            "Longitude": float(rng.uniform(5, 15)),
            "Eunis_lvl1": eunis_letters[i % len(eunis_letters)],
            "ReSurvey": "Y" if i % 3 == 0 else "N",
            "Altitude": float(rng.uniform(0, 2000)),
        })
    # Inject a few NAs in the categorical covariate
    rows[1]["ReSurvey"] = ""
    rows[7]["ReSurvey"] = "NA"
    _write_csv(rows, header_path)

    species_rows = []
    for i in range(50):
        for j in range(rng.integers(3, 8)):
            species_rows.append({
                "PlotObservationID": f"P{i:04d}",
                "WFO_TAXON": f"sp_{j}",
                "Cover %": float(rng.exponential(5.0)),
            })
    _write_csv(species_rows, species_path)
    return header_path, species_path


class TestFromFastCsvCategorical:
    def test_letter_target_and_yn_covariate(self, synthetic_csv_paths):
        header_path, species_path = synthetic_csv_paths

        ds = ResolveDataset.from_fast_csv(
            header=header_path,
            species=species_path,
            roles={
                "plot_id": "PlotObservationID",
                "species_id": "WFO_TAXON",
                "species_plot_id": "PlotObservationID",
                "coords_lat": "Latitude",
                "coords_lon": "Longitude",
                "abundance": "Cover %",
                "covariates": ["Altitude", "ReSurvey"],
            },
            targets={
                "eunis": {"column": "Eunis_lvl1", "task": "classification"},
            },
            categorical_covariates={"ReSurvey": {"Y": 1, "N": 0}},
            verbose=False,
        )

        # Target encoded to int with the EUNIS-natural alphabetical mapping
        eunis_map = ds.categorical_mappings["Eunis_lvl1"]
        assert eunis_map == {
            "M": 0, "N": 1, "P": 2, "Q": 3, "R": 4,
            "S": 5, "T": 6, "U": 7, "V": 8,
        }
        assert ds.targets["eunis"].num_classes == 9

        # Covariate encoded with the explicit mapping
        resurvey_map = ds.categorical_mappings["ReSurvey"]
        assert resurvey_map == {"Y": 1, "N": 0}

        # Header column dtypes
        assert ds.header.schema["Eunis_lvl1"] == pl.Int64
        assert ds.header.schema["ReSurvey"] == pl.Int64

        # Target values reachable via get_target (no nulls remain after target filter)
        targets = ds.get_target("eunis")
        assert targets.dtype.kind == "i"
        assert (targets >= 0).all() and (targets < 9).all()

    def test_auto_yn_covariate(self, synthetic_csv_paths):
        # Pass None for the mapping: auto-build N=0, Y=1 (sorted unique)
        header_path, species_path = synthetic_csv_paths

        ds = ResolveDataset.from_fast_csv(
            header=header_path,
            species=species_path,
            roles={
                "plot_id": "PlotObservationID",
                "species_id": "WFO_TAXON",
                "species_plot_id": "PlotObservationID",
                "coords_lat": "Latitude",
                "coords_lon": "Longitude",
                "abundance": "Cover %",
                "covariates": ["Altitude", "ReSurvey"],
            },
            targets={
                "eunis": {"column": "Eunis_lvl1", "task": "classification"},
            },
            categorical_covariates={"ReSurvey": None},
            verbose=False,
        )

        assert ds.categorical_mappings["ReSurvey"] == {"N": 0, "Y": 1}

    def test_categorical_covariate_must_be_in_roles(self, synthetic_csv_paths):
        header_path, species_path = synthetic_csv_paths

        with pytest.raises(ValueError, match="must also be listed in roles"):
            ResolveDataset.from_fast_csv(
                header=header_path,
                species=species_path,
                roles={
                    "plot_id": "PlotObservationID",
                    "species_id": "WFO_TAXON",
                    "species_plot_id": "PlotObservationID",
                    "coords_lat": "Latitude",
                    "coords_lon": "Longitude",
                    "abundance": "Cover %",
                    "covariates": ["Altitude"],  # ReSurvey not listed
                },
                targets={"eunis": {"column": "Eunis_lvl1", "task": "classification"}},
                categorical_covariates={"ReSurvey": None},
                verbose=False,
            )
