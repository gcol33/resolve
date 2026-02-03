"""Debug CUDA hash state."""
import sys
sys.path.insert(0, 'src/core/python/src')
from resolve_core import ResolveDataset, DatasetConfig, RoleMapping, TargetSpec, SpeciesEncodingMode

# Create config with cuda hash
config = DatasetConfig()
config.species_encoding = SpeciesEncodingMode.Hash
config.hash_dim = 512
config.use_cuda_hash = True

# Load a small test dataset
roles = RoleMapping()
roles.plot_id = 'PlotObservationID'
roles.species_id = 'WFO_TAXON'
roles.abundance = 'Cover %'
roles.longitude = 'Longitude'
roles.latitude = 'Latitude'
roles.genus = 'WFO_GENUS'
roles.family = 'WFO_FAMILY'

targets = [TargetSpec.regression('Relevé area (m²)')]

print("Loading dataset with use_cuda_hash=True...")
dataset = ResolveDataset.from_csv(
    'J:/Phd Local/Gilles_paper_resolve/data/iter_bench_header.csv',
    'J:/Phd Local/Gilles_paper_resolve/data/iter_bench_species.csv',
    roles,
    targets,
    config
)

print('has_raw_species_data:', dataset.has_raw_species_data())
he = dataset.hash_embedding()
print('hash_embedding defined:', he.defined() if hasattr(he, 'defined') else 'N/A')
print('hash_embedding numel:', he.numel())
print('hash_embedding shape:', tuple(he.shape) if he.numel() > 0 else 'empty')

rsi = dataset.raw_species_ids()
print('raw_species_ids defined:', rsi.defined() if hasattr(rsi, 'defined') else 'N/A')
print('raw_species_ids numel:', rsi.numel())

rw = dataset.raw_weights()
print('raw_weights defined:', rw.defined() if hasattr(rw, 'defined') else 'N/A')
print('raw_weights numel:', rw.numel())

po = dataset.plot_offsets()
print('plot_offsets defined:', po.defined() if hasattr(po, 'defined') else 'N/A')
print('plot_offsets numel:', po.numel())
