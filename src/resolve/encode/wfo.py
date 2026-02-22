"""Local WFO (World Flora Online) backbone for fast species name matching.

Downloads and indexes the WFO Plant List backbone taxonomy, then provides
exact and fuzzy species name matching with synonym resolution. Results can
be converted to a TaxonomyNormalizer for use in RESOLVE training/inference.

The WFO backbone (~1.4M names, ~118 MB compressed) is downloaded from Zenodo
and stored locally. Matching is done entirely offline.

Usage:
    # Download backbone (one-time, ~118 MB)
    backbone = WFOBackbone.download("data/wfo")

    # Or load an existing backbone
    backbone = WFOBackbone("data/wfo/classification.txt")

    # Match species names
    result = backbone.match_one("Quercus robur")
    results = backbone.match_batch(["Quercus robur", "Pinus sylvestris"])

    # Build a TaxonomyNormalizer for training
    normalizer = backbone.to_normalizer(my_species_list)
"""

from __future__ import annotations

import io
import re
import zipfile
from difflib import get_close_matches
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Name preparation (equivalent to R WorldFlora's preprocessing options)
# ---------------------------------------------------------------------------

# Qualifiers that indicate uncertain identification — removed before matching.
# Match qualifier word + optional trailing period (period is NOT a word char,
# so \b won't capture it — we explicitly include \.? outside the boundary).
_QUALIFIERS = re.compile(
    r"\b(?:cf|aff|s\.l|s\.str|sp|spp|subsp|var|f|"
    r"auct|sensu|non|nec|vel|agg)\.?"
    r"(?=\s|$)",
    re.IGNORECASE,
)

# Authorship patterns: capital letter(s) followed by optional period at end of name,
# or parenthesized author citations like "(L.)" or "(Aiton) Sm."
_AUTHOR_PARENS = re.compile(r"\([A-Z][a-zé.&\s]*\)")
_AUTHOR_TRAILING = re.compile(
    r"\s+[A-Z][a-zé.]*\.?"           # First author token (e.g., "L." or "Sm.")
    r"(?:\s+(?:ex|in|&)\s+[A-Z][a-zé.]*\.?)*"  # Additional authors
    r"$"
)

# Brackets and numbers
_BRACKETS = re.compile(r"\([^)]*\)")
_NUMBERS = re.compile(r"\d+")
_MULTI_SPACE = re.compile(r"\s+")


def prepare_name(
    name: str,
    remove_qualifiers: bool = True,
    remove_authorship: bool = True,
    remove_brackets: bool = True,
    remove_numbers: bool = True,
    lowercase_epithet: bool = True,
) -> str:
    """Clean a species name for matching against a taxonomy backbone.

    Mirrors the preprocessing in R WorldFlora's WFO.match():
    - Removes taxonomic qualifiers (cf., aff., sp., var., etc.)
    - Removes authorship citations (L., (Aiton) Sm., etc.)
    - Removes parenthesized content
    - Removes numbers
    - Lowercases specific epithets (keeps genus capitalized)

    Args:
        name: Raw species name string.
        remove_qualifiers: Strip cf., aff., sp., etc.
        remove_authorship: Strip trailing author citations.
        remove_brackets: Strip parenthesized content.
        remove_numbers: Strip numeric characters.
        lowercase_epithet: Lowercase everything except the genus.

    Returns:
        Cleaned species name suitable for backbone matching.

    Examples:
        >>> prepare_name("Quercus robur L.")
        'Quercus robur'
        >>> prepare_name("Pinus cf. sylvestris")
        'Pinus sylvestris'
        >>> prepare_name("Rosa canina var. dumalis (Bechst.) Baker")
        'Rosa canina dumalis'
    """
    if not name or not isinstance(name, str):
        return name

    s = name.strip()

    if remove_brackets:
        s = _BRACKETS.sub(" ", s)

    if remove_qualifiers:
        s = _QUALIFIERS.sub(" ", s)

    if remove_numbers:
        s = _NUMBERS.sub(" ", s)

    if remove_authorship:
        s = _AUTHOR_PARENS.sub(" ", s)
        s = _AUTHOR_TRAILING.sub("", s)

    # Collapse whitespace
    s = _MULTI_SPACE.sub(" ", s).strip()

    if lowercase_epithet and s:
        # Keep genus (first word) as-is, lowercase the rest
        parts = s.split(" ", 1)
        if len(parts) == 2:
            s = parts[0] + " " + parts[1].lower()

    return s


def prepare_names(names: list[str], **kwargs) -> list[str]:
    """Clean a list of species names. See prepare_name() for options."""
    return [prepare_name(n, **kwargs) for n in names]


# Zenodo URLs for WFO backbone (R-compatible Darwin Core archive)
WFO_ZENODO_URLS = {
    "2024-12": "https://zenodo.org/records/14538251/files/_DwC_backbone_R.zip",
    "2024-06": "https://zenodo.org/records/12171908/files/_DwC_backbone_R.zip",
}

# Columns we actually need from the ~25-column classification.txt
_USED_COLUMNS = [
    "taxonID",
    "scientificName",
    "taxonRank",
    "acceptedNameUsageID",
    "taxonomicStatus",
    "family",
    "genus",
    "specificEpithet",
]


class WFOBackbone:
    """Local WFO backbone for fast species name matching.

    Loads the WFO classification.txt (Darwin Core Archive) and builds
    in-memory indices for O(1) exact lookups and optional fuzzy matching.

    Matching follows the same logic as the R WorldFlora package:
    1. Exact match on scientificName
    2. Optional fuzzy match via Levenshtein distance
    3. Synonym resolution via acceptedNameUsageID
    """

    def __init__(self, classification_path: str | Path):
        """Load WFO backbone from classification.txt.

        Args:
            classification_path: Path to classification.txt
                (from _DwC_backbone_R.zip, tab-separated)
        """
        self._path = Path(classification_path)
        if not self._path.exists():
            raise FileNotFoundError(f"WFO backbone not found: {self._path}")

        self._df = self._load_backbone(self._path)
        self._name_index: dict[str, list[int]] = {}
        self._id_index: dict[str, int] = {}
        self._build_indices()

    @staticmethod
    def _load_backbone(path: Path) -> pl.DataFrame:
        """Load classification.txt efficiently (only needed columns)."""
        df = pl.read_csv(
            path,
            separator="\t",
            encoding="utf8",
            columns=_USED_COLUMNS,
            schema_overrides={
                "taxonID": pl.Utf8,
                "scientificName": pl.Utf8,
                "taxonRank": pl.Utf8,
                "acceptedNameUsageID": pl.Utf8,
                "taxonomicStatus": pl.Utf8,
                "family": pl.Utf8,
                "genus": pl.Utf8,
                "specificEpithet": pl.Utf8,
            },
            infer_schema_length=0,  # treat all as string
        )
        # Normalize whitespace
        strip_cols = [c for c in ("scientificName", "family", "genus", "specificEpithet") if c in df.columns]
        if strip_cols:
            df = df.with_columns(
                [pl.col(c).str.strip_chars().alias(c) for c in strip_cols]
            )
        return df

    def _build_indices(self) -> None:
        """Build lookup dicts for fast matching."""
        names = self._df["scientificName"].to_list()
        ids = self._df["taxonID"].to_list()

        for i in range(len(self._df)):
            name = names[i]
            if isinstance(name, str):
                if name not in self._name_index:
                    self._name_index[name] = []
                self._name_index[name].append(i)

            tid = ids[i]
            if isinstance(tid, str):
                self._id_index[tid] = i

    @property
    def n_names(self) -> int:
        """Total number of name records in the backbone."""
        return len(self._df)

    @property
    def n_accepted(self) -> int:
        """Number of accepted name records."""
        return self._df.filter(pl.col("taxonomicStatus") == "ACCEPTED").height

    @property
    def n_synonyms(self) -> int:
        """Number of synonym records."""
        return self._df.filter(
            pl.col("taxonomicStatus").str.contains("SYNONYM")
        ).height

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    @classmethod
    def download(
        cls,
        dest_dir: str | Path,
        version: str = "2024-12",
        verbose: bool = True,
    ) -> WFOBackbone:
        """Download WFO backbone from Zenodo, extract, and return loaded backbone.

        Args:
            dest_dir: Directory to store the extracted classification.txt.
            version: WFO release version ("2024-12" or "2024-06").
            verbose: Print progress messages.

        Returns:
            Loaded WFOBackbone instance.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        classification_path = dest_dir / "classification.txt"

        # Skip download if already exists
        if classification_path.exists():
            if verbose:
                size_mb = classification_path.stat().st_size / (1024 * 1024)
                print(f"WFO backbone already exists: {classification_path} ({size_mb:.0f} MB)")
            return cls(classification_path)

        url = WFO_ZENODO_URLS.get(version)
        if url is None:
            available = ", ".join(sorted(WFO_ZENODO_URLS.keys()))
            raise ValueError(f"Unknown WFO version {version!r}. Available: {available}")

        zip_path = dest_dir / f"wfo_backbone_{version}.zip"

        if verbose:
            print(f"Downloading WFO backbone {version} from Zenodo...")
            print(f"  URL: {url}")
            print(f"  Destination: {zip_path}")

        # Download
        def _progress_hook(count, block_size, total_size):
            if total_size > 0 and count % 100 == 0:
                pct = count * block_size / total_size * 100
                print(f"\r  {pct:.0f}%", end="", flush=True)

        urlretrieve(url, zip_path, reporthook=_progress_hook if verbose else None)
        if verbose:
            print()

        # Extract classification.txt
        if verbose:
            print(f"Extracting classification.txt...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find classification.txt in the archive
            txt_names = [n for n in zf.namelist() if n.endswith("classification.txt")]
            if not txt_names:
                raise FileNotFoundError(
                    f"classification.txt not found in {zip_path}. "
                    f"Contents: {zf.namelist()[:10]}"
                )
            # Extract to dest_dir
            with zf.open(txt_names[0]) as src:
                classification_path.write_bytes(src.read())

        size_mb = classification_path.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"Extracted: {classification_path} ({size_mb:.0f} MB)")

        # Clean up zip
        zip_path.unlink()
        if verbose:
            print(f"Removed zip: {zip_path}")

        return cls(classification_path)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _get_row(self, idx: int) -> dict:
        """Get a row from the backbone as a dict (helper for iloc-like access)."""
        return self._df.row(idx, named=True)

    def _resolve_accepted(self, row_idx: int) -> dict:
        """Resolve a matched row to its accepted name (following synonyms)."""
        row = self._get_row(row_idx)
        status = str(row.get("taxonomicStatus") or "")
        accepted_id = row.get("acceptedNameUsageID")

        result = {
            "matched_name": row["scientificName"],
            "matched_id": row["taxonID"],
            "matched_status": status,
            "family": row.get("family"),
            "genus": row.get("genus"),
        }

        # If this is a synonym, resolve to accepted name
        if "SYNONYM" in status and accepted_id is not None and str(accepted_id) in self._id_index:
            acc_idx = self._id_index[str(accepted_id)]
            acc_row = self._get_row(acc_idx)
            result["accepted_name"] = acc_row["scientificName"]
            result["accepted_id"] = acc_row["taxonID"]
            result["accepted_status"] = str(acc_row.get("taxonomicStatus") or "")
            result["family"] = acc_row.get("family") or result["family"]
            result["genus"] = acc_row.get("genus") or result["genus"]
            result["was_synonym"] = True
        elif status == "ACCEPTED":
            result["accepted_name"] = row["scientificName"]
            result["accepted_id"] = row["taxonID"]
            result["accepted_status"] = status
            result["was_synonym"] = False
        else:
            # UNCHECKED, DOUBTFUL, etc. — use as-is
            result["accepted_name"] = row["scientificName"]
            result["accepted_id"] = row["taxonID"]
            result["accepted_status"] = status
            result["was_synonym"] = False

        return result

    def _pick_best(self, row_indices: list[int]) -> int:
        """Pick the best match from multiple candidates (like WFO.one).

        Priority:
        1. ACCEPTED over SYNONYM
        2. SPECIES rank over higher ranks
        3. Smallest taxonID as tiebreaker
        """
        if len(row_indices) == 1:
            return row_indices[0]

        candidates = []
        for idx in row_indices:
            row = self._get_row(idx)
            status = str(row.get("taxonomicStatus") or "")
            rank = str(row.get("taxonRank") or "")
            tid = str(row.get("taxonID") or "")
            candidates.append((
                0 if status == "ACCEPTED" else 1,  # Prefer accepted
                0 if rank == "SPECIES" else 1,      # Prefer species rank
                tid,                                  # Tiebreak by ID
                idx,
            ))
        candidates.sort()
        return candidates[0][3]

    def match_one(
        self,
        name: str,
        fuzzy: float = 0.0,
        fuzzy_limit: int = 5,
        prepare: bool = True,
    ) -> Optional[dict]:
        """Match a single species name against the WFO backbone.

        Args:
            name: Species name to match.
            fuzzy: Maximum edit distance ratio for fuzzy matching (0 = exact only,
                   0.1 = 10% of string length, following R WorldFlora convention).
            fuzzy_limit: Max number of fuzzy candidates to consider.
            prepare: If True, clean the name before matching (remove authorship,
                     qualifiers, brackets, numbers). See prepare_name().

        Returns:
            Match result dict with keys: input, accepted_name, accepted_id,
            matched_name, matched_id, matched_status, family, genus,
            was_synonym, match_method, fuzzy_dist.
            Returns None if no match found.
        """
        original = name.strip()

        # Try raw name first, then prepared name
        candidates = [original]
        if prepare:
            prepared = prepare_name(original)
            if prepared != original:
                candidates.append(prepared)

        for candidate in candidates:
            # Step 1: Exact match
            if candidate in self._name_index:
                best_idx = self._pick_best(self._name_index[candidate])
                result = self._resolve_accepted(best_idx)
                result["input"] = original
                result["match_method"] = "exact" if candidate == original else "exact_prepared"
                result["fuzzy_dist"] = 0
                return result

            # Step 2: Case-insensitive exact match
            candidate_lower = candidate.lower()
            for backbone_name, indices in self._name_index.items():
                if backbone_name.lower() == candidate_lower:
                    best_idx = self._pick_best(indices)
                    result = self._resolve_accepted(best_idx)
                    result["input"] = original
                    result["match_method"] = "exact_ci" if candidate == original else "exact_ci_prepared"
                    result["fuzzy_dist"] = 0
                    return result

        # Step 3: Fuzzy match on prepared name (if enabled)
        if fuzzy > 0:
            search_name = candidates[-1]  # Use prepared name for fuzzy
            return self._match_fuzzy(search_name, fuzzy, fuzzy_limit, original_input=original)

        return None

    def _match_fuzzy(
        self,
        name: str,
        cutoff: float,
        n: int,
        original_input: Optional[str] = None,
    ) -> Optional[dict]:
        """Fuzzy match using edit distance.

        Uses rapidfuzz if available, falls back to difflib.
        """
        input_label = original_input or name

        # Try rapidfuzz first (much faster for large vocabularies)
        try:
            from rapidfuzz import process, fuzz
            matches = process.extract(
                name,
                self._name_index.keys(),
                scorer=fuzz.ratio,
                limit=n,
            )
            # rapidfuzz scores are 0-100, convert cutoff
            min_score = (1.0 - cutoff) * 100
            matches = [(m, score, key) for m, score, key in matches if score >= min_score]
            if matches:
                best_name = matches[0][0]
                best_score = matches[0][1]
                best_idx = self._pick_best(self._name_index[best_name])
                result = self._resolve_accepted(best_idx)
                result["input"] = input_label
                result["match_method"] = "fuzzy"
                result["fuzzy_dist"] = round(1.0 - best_score / 100, 3)
                return result
            return None
        except ImportError:
            pass

        # Fallback: difflib.get_close_matches (slower but stdlib)
        close = get_close_matches(name, self._name_index.keys(), n=n, cutoff=1.0 - cutoff)
        if close:
            best_name = close[0]
            best_idx = self._pick_best(self._name_index[best_name])
            result = self._resolve_accepted(best_idx)
            result["input"] = input_label
            result["match_method"] = "fuzzy"
            # Compute approximate edit distance ratio
            max_len = max(len(name), len(best_name))
            if max_len > 0:
                from difflib import SequenceMatcher
                ratio = SequenceMatcher(None, name, best_name).ratio()
                result["fuzzy_dist"] = round(1.0 - ratio, 3)
            else:
                result["fuzzy_dist"] = 0
            return result

        return None

    def match_batch(
        self,
        names: list[str],
        fuzzy: float = 0.0,
        fuzzy_limit: int = 5,
        prepare: bool = True,
        verbose: bool = True,
    ) -> dict[str, Optional[dict]]:
        """Match a batch of species names.

        Args:
            names: List of species names to match.
            fuzzy: Fuzzy matching threshold (0 = exact only).
            fuzzy_limit: Max fuzzy candidates per name.
            prepare: If True, clean names before matching (see prepare_name()).
            verbose: Print progress.

        Returns:
            Dict mapping each input name to its match result (or None).
        """
        results = {}
        n_exact = 0
        n_fuzzy = 0
        n_miss = 0

        for i, name in enumerate(names):
            if verbose and (i + 1) % 5000 == 0:
                print(f"  Matched {i + 1:,}/{len(names):,} species...", flush=True)

            result = self.match_one(name, fuzzy=fuzzy, fuzzy_limit=fuzzy_limit, prepare=prepare)
            results[name] = result

            if result is None:
                n_miss += 1
            elif "fuzzy" in result.get("match_method", ""):
                n_fuzzy += 1
            else:
                n_exact += 1

        if verbose:
            print(f"  Matched {len(names):,} species: "
                  f"{n_exact:,} exact, {n_fuzzy:,} fuzzy, {n_miss:,} unmatched")

        return results

    def to_normalizer(
        self,
        species_names: list[str],
        fuzzy: float = 0.0,
        verbose: bool = True,
    ):
        """Create a TaxonomyNormalizer from matching results.

        Matches all provided species names against the backbone and builds
        a normalizer mapping original → accepted canonical name.

        Args:
            species_names: Unique species names from the dataset.
            fuzzy: Fuzzy matching threshold.
            verbose: Print progress.

        Returns:
            TaxonomyNormalizer with {original_name: accepted_name} mapping.
        """
        from resolve.encode.normalize import TaxonomyNormalizer

        results = self.match_batch(species_names, fuzzy=fuzzy, verbose=verbose)

        mapping = {}
        for name, result in results.items():
            if result is not None and result.get("accepted_name"):
                mapping[name] = result["accepted_name"]
            else:
                mapping[name] = name  # Identity for unmatched

        return TaxonomyNormalizer(mapping, backbone="wfo")

    def to_normalizer_with_taxonomy(
        self,
        species_names: list[str],
        fuzzy: float = 0.0,
        verbose: bool = True,
    ) -> tuple:
        """Create normalizer + taxonomy lookup from matching results.

        Returns both a TaxonomyNormalizer and species→genus/family dicts,
        useful for bag-of-species encoding where taxonomy comes from the
        backbone rather than the dataset.

        Args:
            species_names: Unique species names from the dataset.
            fuzzy: Fuzzy matching threshold.
            verbose: Print progress.

        Returns:
            (normalizer, species_to_genus, species_to_family) tuple.
        """
        from resolve.encode.normalize import TaxonomyNormalizer

        results = self.match_batch(species_names, fuzzy=fuzzy, verbose=verbose)

        mapping = {}
        species_to_genus = {}
        species_to_family = {}

        for name, result in results.items():
            if result is not None:
                accepted = result.get("accepted_name", name)
                mapping[name] = accepted
                genus = result.get("genus")
                family = result.get("family")
                if genus and isinstance(genus, str):
                    species_to_genus[accepted] = genus
                if family and isinstance(family, str):
                    species_to_family[accepted] = family
            else:
                mapping[name] = name

        normalizer = TaxonomyNormalizer(mapping, backbone="wfo")
        return normalizer, species_to_genus, species_to_family

    def __repr__(self) -> str:
        return (
            f"WFOBackbone(path={self._path.name!r}, "
            f"n_names={self.n_names:,}, n_accepted={self.n_accepted:,}, "
            f"n_synonyms={self.n_synonyms:,})"
        )
