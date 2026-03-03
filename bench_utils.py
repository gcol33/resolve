"""Shared utilities for RESOLVE benchmarks."""

import sys
sys.path.insert(0, 'src/core/python/src')
sys.path.insert(0, 'src')

from resolve_core import RoleMapping  # noqa: E402


HEADER_PATH = "J:/Phd Local/Gilles_paper_resolve/data/iter_bench_header.csv"
SPECIES_PATH = "J:/Phd Local/Gilles_paper_resolve/data/iter_bench_species.csv"


def create_role_mapping() -> RoleMapping:
    """Create the standard ASAAS role mapping used by all benchmarks."""
    roles = RoleMapping()
    roles.plot_id = "PlotObservationID"
    roles.species_id = "WFO_TAXON"
    roles.abundance = "Cover %"
    roles.longitude = "Longitude"
    roles.latitude = "Latitude"
    roles.genus = "WFO_GENUS"
    roles.family = "WFO_FAMILY"
    return roles
