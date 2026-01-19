"""Semantic role definitions for Spacc datasets."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RoleMapping:
    """
    Maps semantic roles to column names in user data.

    Required roles:
        plot_id: Unique identifier for each plot observation
        species_id: Species identifier in species table
        species_plot_id: Foreign key linking species rows to plots
        coords_lat: Latitude coordinate
        coords_lon: Longitude coordinate

    Optional roles:
        abundance: Cover percentage or count (if absent, presence-absence assumed)
        taxonomy_genus: Genus name for learned embeddings
        taxonomy_family: Family name for learned embeddings
        covariates: List of additional continuous feature columns
    """

    # Required
    plot_id: str
    species_id: str
    species_plot_id: str
    coords_lat: str
    coords_lon: str

    # Optional
    abundance: Optional[str] = None
    taxonomy_genus: Optional[str] = None
    taxonomy_family: Optional[str] = None
    covariates: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Check that required roles are specified."""
        required = ["plot_id", "species_id", "species_plot_id", "coords_lat", "coords_lon"]
        missing = [r for r in required if not getattr(self, r)]
        if missing:
            raise ValueError(f"Missing required roles: {missing}")

    @property
    def has_abundance(self) -> bool:
        return self.abundance is not None

    @property
    def has_taxonomy(self) -> bool:
        return self.taxonomy_genus is not None and self.taxonomy_family is not None

    @classmethod
    def from_dict(cls, mapping: dict[str, str | list[str]]) -> "RoleMapping":
        """Create RoleMapping from a dictionary."""
        return cls(
            plot_id=mapping["plot_id"],
            species_id=mapping["species_id"],
            species_plot_id=mapping["species_plot_id"],
            coords_lat=mapping["coords_lat"],
            coords_lon=mapping["coords_lon"],
            abundance=mapping.get("abundance"),
            taxonomy_genus=mapping.get("taxonomy_genus"),
            taxonomy_family=mapping.get("taxonomy_family"),
            covariates=mapping.get("covariates", []),
        )


@dataclass
class TargetConfig:
    """Configuration for a prediction target."""

    column: str
    task: str  # "regression" or "classification"
    transform: Optional[str] = None  # "log1p" or None
    num_classes: Optional[int] = None  # for classification
    weight: float = 1.0  # loss weight in multi-task

    def __post_init__(self):
        if self.task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got '{self.task}'")
        if self.task == "classification" and self.num_classes is None:
            raise ValueError("num_classes required for classification tasks")
        if self.transform is not None and self.transform not in ("log1p",):
            raise ValueError(f"Unknown transform: {self.transform}")

    @classmethod
    def from_dict(cls, name: str, config: dict) -> "TargetConfig":
        return cls(
            column=config["column"],
            task=config["task"],
            transform=config.get("transform"),
            num_classes=config.get("num_classes"),
            weight=config.get("weight", 1.0),
        )
